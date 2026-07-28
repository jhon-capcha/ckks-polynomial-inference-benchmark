"""
Arquitectura CNN base del proyecto.

Implementa una LeNet reducida compatible con el diseño experimental:

    Entrada 1x28x28
    -> Conv1(1->6, kernel=5, padding=2)
    -> Activación 1
    -> AvgPool2d
    -> Conv2(6->16, kernel=5)
    -> Activación 2
    -> AvgPool2d
    -> Flatten
    -> Linear(400->120)
    -> Activación 3
    -> Linear(120->10)

La activación se recibe mediante una factory para permitir sustituir ReLU por
activaciones polinómicas sin modificar la arquitectura.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn

ActivationFactory = Callable[[], nn.Module]


class ReducedLeNet(nn.Module):
    """LeNet reducida para clasificación de MNIST.

    Parameters
    ----------
    activation_factory:
        Callable sin argumentos que construye un módulo de activación.

        Por defecto se crean tres instancias independientes de ``nn.ReLU``.
        Posteriormente podrá utilizarse una factory de activaciones
        polinómicas para las evaluaciones en texto plano.

    Notes
    -----
    La salida del modelo contiene logits. No se aplica softmax porque
    ``nn.CrossEntropyLoss`` espera logits sin normalizar.
    """

    def __init__(
        self,
        activation_factory: ActivationFactory = nn.ReLU,
    ) -> None:
        super().__init__()

        # Bloques convolucionales.
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
            padding=2,
        )
        self.act1 = activation_factory()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            padding=0,
        )
        self.act2 = activation_factory()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Clasificador.
        # Después del segundo pooling:
        # 16 canales x 5 x 5 = 400 características.
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=400, out_features=120)
        self.act3 = activation_factory()
        self.fc2 = nn.Linear(in_features=120, out_features=10)

    def forward(self, x: Tensor) -> Tensor:
        """Ejecuta la propagación hacia adelante.

        Parameters
        ----------
        x:
            Tensor con forma ``(batch_size, 1, 28, 28)``.

        Returns
        -------
        Tensor
            Logits con forma ``(batch_size, 10)``.

        Raises
        ------
        ValueError
            Si la entrada no tiene la forma esperada para imágenes MNIST.
        """
        self._validate_input(x)

        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.act3(x)
        logits = self.fc2(x)

        return logits

    @staticmethod
    def _validate_input(x: Tensor) -> None:
        """Valida que la entrada corresponda al formato MNIST."""
        if x.ndim != 4:
            raise ValueError(
                "La entrada debe tener cuatro dimensiones: "
                "(batch_size, channels, height, width). "
                f"Forma recibida: {tuple(x.shape)}"
            )

        expected_shape = (1, 28, 28)

        if tuple(x.shape[1:]) != expected_shape:
            raise ValueError(
                "La entrada debe tener forma "
                f"(batch_size, {expected_shape[0]}, "
                f"{expected_shape[1]}, {expected_shape[2]}). "
                f"Forma recibida: {tuple(x.shape)}"
            )


def count_trainable_parameters(model: nn.Module) -> int:
    """Devuelve el número de parámetros entrenables del modelo."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


if __name__ == "__main__":
    # Prueba manual mínima para validar dimensiones.
    model = ReducedLeNet()
    sample = torch.randn(2, 1, 28, 28)
    output = model(sample)

    print(model)
    print(f"Forma de entrada: {tuple(sample.shape)}")
    print(f"Forma de salida: {tuple(output.shape)}")
    print(f"Parámetros entrenables: {count_trainable_parameters(model):,}")
