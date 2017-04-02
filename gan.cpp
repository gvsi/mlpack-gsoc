// Includes all relevant components of mlpack.
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/elu.hpp>
#include <mlpack/core/optimizers/rmsprop/rmsprop.hpp>

// Convenience
using namespace mlpack;
using namespace mlpack::distribution;
using namespace mlpack::ann;

int main()
{
  // Original data
  GaussianDistribution g1("4", "1.25");

  // Generator
  int input_size = 1;
  int hidden_size = 50;
  int output_size = 1;

  FFN<NegativeLogLikelihood<> > generator;
  generator.Add<Linear<> >(input_size, hidden_size);
  generator.Add<ELU<> >();
  generator.Add<Linear<> >(hidden_size, hidden_size);
  generator.Add<SigmoidLayer<> >();
  generator.Add<Linear<> >(hidden_size, output_size);

  // Discriminator
  FFN<NegativeLogLikelihood<> > discriminator;
  discriminator.Add<Linear<> >(input_size, hidden_size);
  discriminator.Add<ELU<> >();
  discriminator.Add<Linear<> >(hidden_size, hidden_size);
  discriminator.Add<ELU<> >();
  discriminator.Add<Linear<> >(hidden_size, output_size);
  discriminator.Add<SigmoidLayer<> >();

  int num_epochs = 30000;

  // Follow original GAN paper size (i.e. train D and G one sample at a time)
  int d_steps = 1;
  int g_steps = 1;

  for (int epoch_index = 0; epoch_index < num_epochs; ++epoch_index) {
    arma::mat label(1, 1);

    // Train Discriminator
    for (int d_index = 0; d_index < d_steps; ++d_index) {

      // Train discriminator on real data
      arma::mat d_input(1, 1);
      d_input(0, 0) = g1.Random()[0];

      arma::mat real_point_pred(1, 1);
      discriminator.Predict(d_input, real_point_pred);

      label(0, 0) = 1; // it's a real value

      // TODO: this should make one forward and one backward pass instead
      discriminator.Train(real_point_pred, label);


      // Train discriminator on fake data
      arma::mat gi_input(1, 1, arma::fill::randu);

      arma::mat fake_point(1, 1);
      generator.Predict(gi_input, fake_point);

      arma::mat fake_point_pred(1, 1);
      discriminator.Predict(fake_point, fake_point_pred);

      label(0, 0) = 0; // it's a fake value

      // TODO: this should make one forward and one backward pass instead
      discriminator.Train(fake_point_pred, label);
    }

    // Train Discriminator
    for (int g_index = 0; g_index < g_steps; ++g_index) {
      arma::mat gi_input(1, 1, arma::fill::randu);

      arma::mat fake_point(1, 1);
      generator.Predict(gi_input, fake_point);

      arma::mat fake_point_pred(1, 1);
      discriminator.Predict(fake_point, fake_point_pred);
      label(0, 0) = 1; // trick generator into thinking it's a real point

      // TODO: this should make one forward and one backward pass instead
      discriminator.Train(fake_point_pred, label);
    }
  }
}
