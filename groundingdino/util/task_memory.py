import torch
import torch.nn.functional as F
import os

def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance

@singleton
class TaskMemory:
    def __init__(self):
        self._current_classes = []
        self._modules = {}
        self._memory = {}
        """
        self._modules = {
            layer_id: {
                class_name: [tensor_A, tensor_B]
            }
        }
        """
        self._registered_instances = []
        self._class_counter = {}
        self.current_task = None
        self.is_warm_up = True
        self.task_index = 0
        self.task_mapping = None
        self.W_dim = 10000
        self.W = None
        self.G = None
        self.C = {}
        self.number = 0
        # Streaming statistics used by the LDA classifier.
        self.stream_class_count = {}
        self.stream_class_mean = {}
        self.stream_cov_M2 = None
        self.stream_feature_dim = None
        self.stream_statistics_dtype = torch.float32
        # 为了统计分数
        self.min_scores = {}
        self.mixure_lora = True

    def get_classes(self):
        assert self._current_classes, 'Please set self._current_classes before getting them'
        return self._current_classes

    def set_classes(self, classes):
        self._current_classes = classes

    def get_modules(self, layer_id):
        if layer_id not in self._modules:
            return {}
        return self._modules[layer_id]

    def set_counter(self, classes):
        if not self._class_counter:
            for c in classes:
                self._class_counter[c] = 0

    def add_counter(self, classes):
        if self.is_warm_up:
            return
        for c in classes:
            self._class_counter[c[6:]] += 1

    def print_counter(self):
        for key in sorted(self._class_counter.keys()):
            print(f"{key}: {self._class_counter[key]}")

    @torch.no_grad()
    def save_modules(self):
        for m in self._registered_instances:
            if m.layer_name not in self._modules:
                self._modules[m.layer_name] = {}
            curr_classes = m.per_class_lora_A.keys()
            for curr_class in curr_classes:
                a_b = [
                    m.per_class_lora_A[curr_class]
                ]
                self._modules[m.layer_name][curr_class] = a_b
            
            if any(k.startswith("B_prev_task") for k in self._memory):
                lambda_b_marouf = 0.7
                prev_task_key = "B_prev_task" + m.layer_name
                assert prev_task_key in self._memory, f"{prev_task_key} non trovato in self._memory"
                m.shared_lora_b.data = (1 - lambda_b_marouf) * self._memory[prev_task_key].data + lambda_b_marouf * m.shared_lora_b.data
            self.set_memory("B_shared", m.layer_name, m.shared_lora_b)
        self._registered_instances = []

    @torch.no_grad()
    def merge_B(self):
        if any(k.startswith("B_prev_task") for k in self._memory):
            lambda_b_marouf = 0.7
            for key in self._memory.keys():
                if key.startswith("B_shared"):
                    suffix = key[len("B_shared"):]
                    prev_task_key = "B_prev_task" + suffix
                    assert prev_task_key in self._memory, f"{prev_task_key} non trovato in self._memory"
                    self._memory[key] = (1 - lambda_b_marouf) * self._memory[prev_task_key] + lambda_b_marouf * self._memory[key]

    def set_memory(self, key, layer_name, value):
        assert isinstance(value, torch.Tensor), f"Value must be a tensor, but got {type(value)}"
        self._memory[key+layer_name] = value

    def get_memory(self, key, layer_name):
        return self._memory.get(key+layer_name, None)


    def end_task(self):
        # self.save_modules()
        # self.print_counter()
        # self.merge_B()
        self.is_warm_up = True
        self.task_index += 1

    def enable_per_class(self):
        self.is_warm_up = False
        for m in self._registered_instances:
            m.enable_per_class(self.current_task)

    def register_module(self, instance):
        self._registered_instances.append(instance)
    
    def update_LDA(self, feat, gt):
        category_dict = self.task_mapping[self.current_task]
        for i in range(len(gt)):
            img_gt = gt[i]                      
            img_feature = feat[i].detach().cpu()
            class_list = [category_dict[c] for c in img_gt]
            self._update_streaming_lda(img_feature, class_list)

    def _update_streaming_lda(self, feature, class_list):
        """Update class means and shared within-class scatter online.

        For each class, the state follows Welford's recurrence:

            mean_n = mean_{n-1} + (x - mean_{n-1}) / n
            M2_n += (x - mean_{n-1}) * (x - mean_n)^T

        The notebook sums this per-class M2 into one shared matrix and
        divides it by the total number of class observations minus one.
        """
        if feature.dim() != 1:
            raise ValueError(f"Expected a 1D feature, got shape {tuple(feature.shape)}")

        feature = feature.to(dtype=self.stream_statistics_dtype)
        feature_dim = feature.numel()
        if self.stream_feature_dim is None:
            self.stream_feature_dim = feature_dim
            self.stream_cov_M2 = torch.zeros(
                (feature_dim, feature_dim),
                dtype=self.stream_statistics_dtype,
            )
        elif self.stream_feature_dim != feature_dim:
            raise ValueError(
                "Feature dimension changed while updating LDA statistics: "
                f"expected {self.stream_feature_dim}, got {feature_dim}"
            )

        # The notebook uses set(class_list), so duplicate labels in one image
        # must not count the same image more than once for a class.
        for class_name in set(class_list):
            old_count = self.stream_class_count.get(class_name, 0)
            if old_count == 0:
                self.stream_class_count[class_name] = 1
                self.stream_class_mean[class_name] = feature.clone()
                continue

            old_mean = self.stream_class_mean[class_name]
            new_count = old_count + 1
            delta = feature - old_mean
            new_mean = old_mean + delta / new_count
            delta_M2 = torch.outer(delta, feature - new_mean)
            self.stream_cov_M2.add_(delta_M2)
            self.stream_class_count[class_name] = new_count
            self.stream_class_mean[class_name] = new_mean

    def get_streaming_lda_statistics(self):
        """Return final class means, shared covariance, and class counts."""
        total_count = sum(self.stream_class_count.values())
        if total_count < 2:
            raise RuntimeError(
                "At least two class observations are required to compute "
                f"a sample covariance; got {total_count}."
            )
        if self.stream_cov_M2 is None:
            raise RuntimeError("LDA streaming statistics have not been initialized.")

        class_mean = {
            class_name: mean.clone()
            for class_name, mean in self.stream_class_mean.items()
        }
        cov_matrix = self.stream_cov_M2 / (total_count - 1)
        return class_mean, cov_matrix.clone(), dict(self.stream_class_count)

    def save_streaming_lda_statistics(self, path, suffix=''):
        """Save the streaming LDA state and finalized statistics."""
        os.makedirs(path, exist_ok=True)
        class_mean, cov_matrix, class_count = self.get_streaming_lda_statistics()
        torch.save(cov_matrix, os.path.join(path, f'cov_matrix_stream{suffix}.pt'))
        torch.save(class_mean, os.path.join(path, f'class_mean_stream{suffix}.pt'))
        torch.save(class_count, os.path.join(path, f'class_count_stream{suffix}.pt'))
        if suffix:
            torch.save(
                self.stream_cov_M2,
                os.path.join(path, f'cov_M2_stream{suffix}.pt'),
            )
    @staticmethod
    def _merge_streaming_lda_states(states):
        """Merge rank-local Welford states using parallel Welford updates."""
        merged_count = {}
        merged_mean = {}
        merged_M2 = None

        for state in states:
            local_count = state['class_count']
            local_mean = state['class_mean']
            local_M2 = state['cov_M2']
            if local_M2 is None:
                continue
            if merged_M2 is None:
                merged_M2 = torch.zeros_like(local_M2)
            merged_M2.add_(local_M2)

            for class_name, count_b in local_count.items():
                if count_b == 0:
                    continue
                mean_b = local_mean[class_name]
                if class_name not in merged_count:
                    merged_count[class_name] = count_b
                    merged_mean[class_name] = mean_b.clone()
                    continue

                count_a = merged_count[class_name]
                mean_a = merged_mean[class_name]
                count = count_a + count_b
                delta = mean_b - mean_a
                merged_M2.add_(
                    torch.outer(delta, delta) * (count_a * count_b / count)
                )
                merged_mean[class_name] = mean_a + delta * (count_b / count)
                merged_count[class_name] = count

        if merged_M2 is None:
            raise RuntimeError('No rank-local LDA statistics were provided.')
        total_count = sum(merged_count.values())
        if total_count < 2:
            raise RuntimeError(
                'At least two class observations are required to compute '
                f'a sample covariance; got {total_count}.'
            )
        return merged_mean, merged_M2 / (total_count - 1), merged_count

    def merge_distributed_lda_statistics(self, path, world_size):
        """Merge rank-local Welford states into global LDA statistics."""
        states = []
        for rank in range(world_size):
            suffix = f'_rank{rank}'
            states.append({
                'class_count': torch.load(
                    os.path.join(path, f'class_count_stream{suffix}.pt'),
                    map_location='cpu',
                ),
                'class_mean': torch.load(
                    os.path.join(path, f'class_mean_stream{suffix}.pt'),
                    map_location='cpu',
                ),
                'cov_M2': torch.load(
                    os.path.join(path, f'cov_M2_stream{suffix}.pt'),
                    map_location='cpu',
                ),
            })

        class_mean, cov_matrix, class_count = self._merge_streaming_lda_states(states)
        torch.save(cov_matrix, os.path.join(path, 'cov_matrix_stream.pt'))
        torch.save(class_mean, os.path.join(path, 'class_mean_stream.pt'))
        torch.save(class_count, os.path.join(path, 'class_count_stream.pt'))

        for rank in range(world_size):
            suffix = f'_rank{rank}'
            for filename in (
                f'cov_matrix_stream{suffix}.pt',
                f'class_mean_stream{suffix}.pt',
                f'class_count_stream{suffix}.pt',
                f'cov_M2_stream{suffix}.pt',
            ):
                os.remove(os.path.join(path, filename))

    
    def update_gram(self, feat, gt):
        assert feat.dim() == 2
        category_dict = self.task_mapping[self.current_task]
        if self.W is None:
            self.W = torch.randn(feat.shape[1], self.W_dim, device=feat.device)
            self.W.requires_grad = False
        
        if self.G is None:
            self.G = torch.zeros(self.W_dim, self.W_dim, device=feat.device, dtype=torch.double)
        
        feat = feat @ self.W
        feat = F.relu(feat)

        for i in range(len(gt)):
            img_gt = gt[i]                      
            img_feature = feat[i].detach()
            for c in img_gt:
                class_name = category_dict[c]
                class_name = class_name[0].lower()+class_name[1:]
                if class_name not in self.C:
                    self.C[class_name] = torch.zeros(self.W_dim, device=feat.device, dtype=torch.double)
                self.C[class_name] += img_feature
        
        gram = feat.T@feat
        self.G+=gram
        self.number+=len(feat)
    
    def save_PAC(self, save_dir):
        self.C['number'] = self.number
        torch.save(self.G, os.path.join(save_dir, 'PAC_G.pt'))
        torch.save(self.C, os.path.join(save_dir, 'PAC_C.pt'))
        torch.save(self.W, os.path.join(save_dir, 'PAC_W.pt'))
