import pytest
import imaging.alpha_matting_trimap as mod

CASES = [
    ("adaptive_band", (1,)),
    ("sample_border", (1,)),
    ("block_fusion", (1,)),
    ("scale_order", (1,)),
    ("trimap_recompute", (1,)),
    ("edge_check", (1,)),
    ("determinism", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("unknown_region", (1,)),
    ("metrics", (1,)),
    ("matte_solve", (1,)),
    ("trimap_build", (1,)),
    ("alpha_refine", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
