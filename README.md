
---

<h1 align="center">
  
  Learn GAN from Scratch

</h1>

# :star: How to train:

This is the hard part. Read carefully

1. Generator: given noise generate fake data -> `generated_fake_data`
2. Have some ground truth data i.e True Labels
3. Check what the discriminator says about the fake data coming from step 1: 
    - `Discriminator(generated_fake_data)` -> Real/Fake i.e `Dis_result_for_Gen_output` (probabilistic output)
4. **Update Generator:** The generator should generate fake data such that they are as close as to real data. So the Discriminator accepts them as Real data. 
    - `Gen_Loss = loss(Dis_result_for_Gen_output, True Label)`
    - `Gen_loss.backward()`
5. **Update Discriminator:** The discriminator should be able to distinguish real and fake data. 
    - `Discriminator(True data)` -> Real/Fake i.e `Dis_result_for_True_data`
    - `Dis_Loss = loss(Dis_result_for_True_data, True Label)`
    - `Dis_loss.backward()`

----

## Reference:

- :rocket: [Build a Super Simple GAN in PyTorch](https://towardsdatascience.com/build-a-super-simple-gan-in-pytorch-54ba349920e4)
  - [code repo](https://github.com/nbertagnolli/pytorch-simple-gan)

---


