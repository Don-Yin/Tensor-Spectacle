# Tensor-Spectacle: Visualizing a Sequence of Tensors
This is a toy project that uses manim to visulize a sequence of tensors.

## Installation:
```
pip install -r requirements.txt
```
Direct pip installation coming soon.

## Quick Start:
To visualize a list of tensors, you can easily create a class as shown below:
```python
class Example(TensorVisualizationScene):
    def construct(self):
        random_shapes = [(randint(1, 2), randint(3, 6), randint(3, 6), randint(3, 6)) for _ in range(3)]
        tensors = [torch.rand(shape) for shape in random_shapes]
        labels = [f"Random Tensor {i}" for i in range(len(tensors))]
        super().construct(tensors=tensors, labels=labels, duration_each=0.8, duration_gap=1, engine="matplotlib")
```

Then run your script, and watch it happen:

## Demo:
- When using the matplotlib as engine:
![demo](https://github.com/Don-Yin/Tensor-Spectacle/blob/36e25d87ca4653a9711b082fc224b13563507745/assets/TensorsDistribution_matplotlib.gif)

- Native mainim cubes (much much slower)
![demo](https://github.com/Don-Yin/Tensor-Spectacle/blob/36e25d87ca4653a9711b082fc224b13563507745/assets/TensorsDistribution_native.gif)

## Example
An example is at ```_test.py```

## License:
This project is licensed under the CC BY-NC License.

## Acknowledgments:
- [Manim Community](https://www.manim.community/)
- [PyTorch](https://pytorch.org/)


## Citations
@misc{Tensor-Spectacle2023,
  author = {Don Yin},
  title = {Tensor-Spectacle: Visualizing a Sequence of Tensors},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/YourUsername/Tensor-Spectacle}},
}
