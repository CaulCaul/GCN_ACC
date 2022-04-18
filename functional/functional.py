
def get_density(M):
    r, c = M.shape
    return len(M.indices) / (c * r)