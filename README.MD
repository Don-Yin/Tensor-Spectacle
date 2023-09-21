# Tensor-Spectacle: Visualizing a Sequence of Tensors
Uses manim to visulize a sequence of tensors.

- When using the matplotlib as engine:
![demo](https://github.com/Don-Yin/Tensor-Spectacle/blob/36e25d87ca4653a9711b082fc224b13563507745/assets/TensorsDistribution_matplotlib.gif)

- Native mainim cubes (much much slower)
![demo](https://github.com/Don-Yin/Tensor-Spectacle/blob/36e25d87ca4653a9711b082fc224b13563507745/assets/TensorsDistribution_native.gif)


## Installation:
```pip install Tensor-Spectacle```

## Requirements
- LaTex installed
- Whatever other packages on which ```manim``` is based

## Quick Start:
To visualize a list of tensors, you can easily create a class as shown below:
```python
from tensorspec.scenes.tensors_distribution import TensorVisualizationScene

# make some random 4D tensors (batch, channel, h, w)
random_shapes = [(randint(1, 2), randint(3, 6), randint(3, 6), randint(3, 6)) for _ in range(3)]
tensors = [torch.rand(shape) for shape in random_shapes]
labels = [f"Random Tensor {i}" for i in range(len(tensors))]

# make scene
class Example(TensorVisualizationScene):
    def construct(self):
        super().construct(tensors=tensors, labels=labels, duration_each=0.8, duration_gap=1, engine="matplotlib")

# render
with tempconfig(
    {
        "use_opengl_renderer": True,
        "disable_caching": True,
        "write_to_movie": True,
        "resolution": "1080p",
        "renderer": "cairo",  # "opengl", "cairo"; opengl is faster but manim community at the moment has some issues
        "fps": 60,
    }
):
    scene = Example()
    scene.render()
```

Then run your script, and watch it happen.

## Note:
At the moment, using a tensor with channel size less than 3 is recommanded.

## Example
An example is at ```_test.py```

## License:
This project is licensed under the CC BY-NC License.

## Acknowledgments:
- [Manim Community](https://www.manim.community/)
- [PyTorch](https://pytorch.org/)

## Contribution
The initial motivation was to visualize the training dynamics of biomimetic neurons. However, I realized this could be useful in many scenarios. I will continue updating this as long as I work with neural networks. Contributions are welcome.

## Citations
```latex
@misc{Tensor-Spectacle2023,
  author = {Don Yin},
  title = {Tensor-Spectacle: Visualizing a Sequence of Tensors},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/YourUsername/Tensor-Spectacle}},
}
```
