def parse_bool(s: str) -> bool:
    if s.strip().lower() in ('1', 'true', 'yes', 'y', 't'):
        return True
    if s.strip().lower() in ('0', 'false', 'no', 'n', 'f'):
        return False
    raise ValueError()
