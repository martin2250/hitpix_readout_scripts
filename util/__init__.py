def parse_bool(s: str) -> bool:
    if s.strip().lower() in ('1', 'true', 'yes', 'y', 't'):
        return True
    if s.strip().lower() in ('0', 'false', 'no', 'n', 'f'):
        return False
    raise ValueError()

def parse_ndrange(mask_str: str, ndim: int) -> tuple[slice, ...]:
    xy_str = mask_str.split(',')
    assert len(xy_str) == ndim
    xy_index = []
    for s in xy_str:
        if s == ':':
            xy_index.append(slice(None))
            continue
        ss = list(map(int, s.split(':')))
        if len(ss) == 1:
            xy_index.append(ss[0])
        elif len(ss) == 2:
            xy_index.append(slice(*ss))
        else:
            raise ValueError(xy_str)
    return tuple(xy_index[::-1]) # reverse
