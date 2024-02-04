import inspect
from itertools import product
from manim import Arrow, Mobject

# combinations = list(
#             product([i.get_right() for i in self.latent_layers], [self.latent_mu.get_left(), self.latent_sigma.get_left()])
#         )
#         self.latent_arrows = [
#             Arrow(*combo, color=BLACK, max_tip_length_to_length_ratio=0, stroke_width=3) for combo in combinations
#         ]
#         self.add(*self.latent_arrows)


# class PairwiseArrows:
#     def __init__(self, *args, **kwargs):
#         self.mobjects_from, self.edge_from = kwargs.get("_from")
#         self.mobjects_to, self.edge_to = kwargs.get("to")
#         assert isinstance(self.mobjects_from, list), "mobjects_from must be a list"
#         assert isinstance(self.mobjects_to, list), "mobjects_to must be a list"
#         # ================== pop kwargs for parent ==================
#         parent_params = inspect.signature(super().__init__).parameters
#         [kwargs.pop(kw) for kw in list(kwargs.keys()) if kw not in parent_params]
#         super().__init__(*args, **kwargs)
#         # ===========================================================s

#     def _make_arrows(self):
#         combinations = list(product(self.mobjects_from, self.mobjects_to))
