import math
from typing import Tuple

import torch
import torch.nn as nn

from src.models import Discriminator, Generator
from src.utils import convert_float_matrix_to_int_list, generate_even_data


def train(
    max_int: int = 128,
    batch_size: int = 16,
    training_steps: int = 500,
    learning_rate: float = 0.001,
    print_output_every_n_steps: int = 10,
):
    """Trains the even GAN

    Args:
        max_int: The maximum integer our dataset goes to.  It is used to set the size of the binary
            lists
        batch_size: The number of examples in a training batch
        training_steps: The number of steps to train on.
        learning_rate: The learning rate for the generator and discriminator
        print_output_every_n_steps: The number of training steps before we print generated output

    Returns:
        generator: The trained generator model
        discriminator: The trained discriminator model
    """
    input_length = int(math.log(max_int, 2))

    # Models
    generator = Generator(input_length)
    discriminator = Discriminator(input_length)

    # Optimizers
    generator_optimizer = torch.optim.Adam(
        generator.parameters(), lr=learning_rate
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=learning_rate
    )

    # loss
    loss = nn.BCELoss()

    for i in range(training_steps):
        # zero the gradients on each iteration
        generator_optimizer.zero_grad()

        # Create noisy input for generator
        # Need float type instead of int
        noise = torch.randint(0, 2, size=(batch_size, input_length)).float()
        generated_data = generator(noise)

        # Generate examples of even real data
        true_labels, true_data = generate_even_data(
            max_int, batch_size=batch_size
        )
        true_labels = torch.tensor(true_labels).float()
        true_data = torch.tensor(true_data).float()

        # Train the generator
        # We invert the labels here and don't train the discriminator because we want the generator
        # to make things the discriminator classifies as true.
        generator_discriminator_out = discriminator(generated_data)
        generator_loss = loss(
            generator_discriminator_out.squeeze(), true_labels
        )
        generator_loss.backward()
        generator_optimizer.step()

        # Train the discriminator on the true/generated data
        discriminator_optimizer.zero_grad()
        true_discriminator_out = discriminator(true_data)
        true_discriminator_loss = loss(
            true_discriminator_out.squeeze(), true_labels
        )

        # add .detach() here think about this
        generator_discriminator_out = discriminator(generated_data.detach())
        generator_discriminator_loss = loss(
            generator_discriminator_out.squeeze(), torch.zeros(batch_size)
        )
        discriminator_loss = (
            true_discriminator_loss + generator_discriminator_loss
        ) / 2
        discriminator_loss.backward()
        discriminator_optimizer.step()
        if i % print_output_every_n_steps == 0:
            output = convert_float_matrix_to_int_list(generated_data)
            even_count = len(list(filter(lambda x: (x % 2 == 0), output)))
            print(f"steps: {i}, output: {output}, even count: {even_count}/16")

    return generator, discriminator


if __name__ == "__main__":
    g, d = train()
