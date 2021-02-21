import math
from typing import Tuple, Dict
import numpy as np

import torch
import torch.nn as nn

from src.models import Discriminator, Generator
from src.utils import convert_float_matrix_to_int_list, generate_even_data

import matplotlib.pyplot as plt

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
        generator.parameters(), lr=0.001
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=0.001
    )

    # loss
    loss = nn.BCELoss()
    gen_loss = []
    dis_loss = []

    for i in range(training_steps):
        # zero the gradients on each iteration
        generator_optimizer.zero_grad()

        # Create noisy input for generator
        # Need float type instead of int
        noise = torch.randint(0, 2, size=(batch_size, input_length)).float()
        generated_data = generator(noise)

        # Generate examples of even real data
        # true labels: [1,1,1,1,1,1,....] i.e all ones
        # true data: [[0,0,0,0,1,0,0],....] i.e binary code for even numbers
        true_labels, true_data = generate_even_data(
            max_int, batch_size=batch_size
        )
        true_labels = torch.tensor(true_labels).float()
        true_data = torch.tensor(true_data).float()

        # Train the generator
        # We invert the labels here and don't train the discriminator because we want the generator
        # to make things the discriminator classifies as true.
        # true labels: [1,1,1,1,....]
        discriminator_out_gen_data = discriminator(generated_data)
        generator_loss = loss(
            discriminator_out_gen_data.squeeze(), true_labels
        )
        gen_loss.append(generator_loss.item())
        generator_loss.backward()
        generator_optimizer.step()

        # Train the discriminator 
        # Teach Discriminator to distinguish true data with true label i.e [1,1,1,1,....]
        discriminator_optimizer.zero_grad()
        discriminator_out_true_data = discriminator(true_data)
        discriminator_loss_true_data = loss(
            discriminator_out_true_data.squeeze(), true_labels
        )

        # add .detach() here think about this
        discriminator_out_fake_data = discriminator(generated_data.detach())
        fake_labels = torch.zeros(batch_size) # [0,0,0,.....]
        discriminator_loss_fake_data = loss(
            discriminator_out_fake_data.squeeze(), fake_labels 
        )
        # total discriminator loss
        discriminator_loss = (
            discriminator_loss_true_data + discriminator_loss_fake_data
        ) / 2
        
        dis_loss.append(discriminator_loss.item())
        
        discriminator_loss.backward()
        discriminator_optimizer.step()
        if i % print_output_every_n_steps == 0:
            output = convert_float_matrix_to_int_list(generated_data)
            even_count = len(list(filter(lambda x: (x % 2 == 0), output)))
            print(f"steps: {i}, output: {output}, even count: {even_count}/16, Gen Loss: {np.round(generator_loss.item(),4)}, Dis Loss: {np.round(discriminator_loss.item(),4)}")

    history = {}
    history['dis_loss'] = dis_loss
    history['gen_loss'] = gen_loss

    return generator, discriminator, history

def plot_loss(loss_history: Dict):
    
    plt.plot(loss_history["dis_loss"], color='blue', linewidth=2, label="dis")
    plt.plot(loss_history["gen_loss"],  color='orange', linewidth=2, label="gen")    
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("GAN Loss curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/loss_curve.png")
    

if __name__ == "__main__":
    g, d, history = train()
    plot_loss(history)
    
