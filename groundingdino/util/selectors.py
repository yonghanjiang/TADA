def select_by_max_gap(results):
    if not results:
        return []
        
    sorted_res = sorted(results, key=lambda x: x['score'], reverse=True)
    scores = [r['score'] for r in sorted_res]

    diffs = [scores[i] - scores[i+1] for i in range(len(scores)-1)]
    if diffs:
        max_gap_idx = diffs.index(max(diffs)) + 1  # +1 表示取前面部分
    else:
        max_gap_idx = len(scores)
    selected_labels = [r['label'] for r in sorted_res[:max_gap_idx]]
    return selected_labels

def select_by_threshold(results, threshold=0.5):
    if not results:
        return []

    # 按 score 降序排列
    sorted_res = sorted(results, key=lambda x: x['score'], reverse=True)

    # 选出 score >= threshold 的类别
    selected_results = [r['label'] for r in sorted_res if r['score'] >= threshold]
    if len(selected_results) == 0:
        selected_results = [ r['label'] for r in sorted_res ]
    
    return selected_results

def select_by_topk(results, k=3):
    if not results:
        return []

    # 按 score 降序排列
    sorted_res = sorted(results, key=lambda x: x['score'], reverse=True)
    # 取前 k 个 label
    topk_results = [r['label'] for r in sorted_res[:k]]

    return topk_results