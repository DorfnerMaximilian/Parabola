#include <array>
#include <cmath>
#include <complex>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

// #########################################################################
// ## Helper Functions
// #########################################################################

// Implementation of the double factorial function
static double doublefactorial(int n) {
  if (n == -1 || n == 0) {
    return 1.0;
  } else {
    return n * doublefactorial(n - 2);
  }
}

// Implementation of the binomial function
static double binom(int n, int k) {
  double result = 1.0;
  for (int i = 1; i <= k; ++i) {
    result *= (n - i + 1) / static_cast<double>(i);
  }
  return result;
}

static double gamma_func(double alpha, int n) {
  double value = 0.0;
  if (n % 2 == 0) {

    value = (doublefactorial(n - 1) * sqrt(M_PI)) /
            (pow(2, 0.5 * static_cast<double>(n)) *
             pow(alpha, 0.5 * static_cast<double>(n) + 0.5));
  }
  return value;
}

static std::complex<double> gq_func(double alpha, double qk, int n) {
  std::complex<double> sum(0.0, 0.0);
  std::complex<double> factor(0.0, 0.5 * qk / alpha);
  double exponential_factor = exp(-1.0 * qk * qk / 4 / alpha);
  for (int k = 0; k <= n; ++k) {
    if (k == n) {
      sum += gamma_func(alpha, n);
    } else {
      sum += gamma_func(alpha, k) * binom(n, k) * pow(factor, n - k);
    }
  }
  sum *= exponential_factor;
  return sum;
}

// #########################################################################
// ## END Helper Functions
// #########################################################################

// #########################################################################
// ## Definition of the different Data structs
// #########################################################################

struct Monomial {
  std::array<int, 3> indices;
  double prefactor;
  // Constructor
  Monomial(int index1, int index2, int index3, double factor)
      : indices{index1, index2, index3}, prefactor(factor) {}
  // Method to increase index1 by 1
  void increaseIndex1() { indices[0]++; }

  // Method to decrease index1 by 1
  void decreaseIndex1() {
    if (indices[0] > 0) {
      indices[0]--;
    }
  }

  // Method to increase index2 by 1
  void increaseIndex2() { indices[1]++; }

  // Method to decrease index2 by 1
  void decreaseIndex2() {
    if (indices[1] > 0) {
      indices[1]--;
    }
  }

  // Method to increase index3 by 1
  void increaseIndex3() { indices[2]++; }

  // Method to decrease index3 by 1
  void decreaseIndex3() {
    if (indices[2] > 0) {
      indices[2]--;
    }
  }
};

static std::unordered_map<std::string, std::vector<Monomial>> getSolidHarmonics() {
  std::unordered_map<std::string, std::vector<Monomial>> cs;
  cs["s"] = {Monomial(0, 0, 0, 0.5 * std::sqrt(1.0 / M_PI))};

  cs["py"] = {Monomial(0, 1, 0, std::sqrt(3. / (4.0 * M_PI)))};
  cs["pz"] = {Monomial(0, 0, 1, std::sqrt(3. / (4.0 * M_PI)))};
  cs["px"] = {Monomial(1, 0, 0, std::sqrt(3. / (4.0 * M_PI)))};

  cs["d-2"] = {Monomial(1, 1, 0, 0.5 * std::sqrt(15. / M_PI))};
  cs["d-1"] = {Monomial(0, 1, 1, 0.5 * std::sqrt(15. / M_PI))};
  cs["d0"] = {Monomial(2, 0, 0, -0.25 * std::sqrt(5. / M_PI)),
              Monomial(0, 2, 0, -0.25 * std::sqrt(5. / M_PI)),
              Monomial(0, 0, 2, 0.5 * std::sqrt(5. / M_PI))};
  cs["d+1"] = {Monomial(1, 0, 1, 0.5 * std::sqrt(15. / M_PI))};
  cs["d+2"] = {Monomial(2, 0, 0, 0.25 * std::sqrt(15. / M_PI)),
               Monomial(0, 2, 0, -0.25 * std::sqrt(15. / M_PI))};

  cs["f-3"] = {Monomial(2, 1, 0, 0.75 * std::sqrt(35. / 2. / M_PI)),
               Monomial(0, 3, 0, -0.25 * std::sqrt(35. / 2. / M_PI))};
  cs["f-2"] = {Monomial(1, 1, 1, 0.5 * std::sqrt(105. / M_PI))};
  cs["f-1"] = {Monomial(0, 1, 2, std::sqrt(21. / 2. / M_PI)),
               Monomial(2, 1, 0, -0.25 * std::sqrt(21. / 2. / M_PI)),
               Monomial(0, 3, 0, -0.25 * std::sqrt(21. / 2. / M_PI))};
  cs["f0"] = {Monomial(0, 0, 3, 0.5 * std::sqrt(7. / M_PI)),
              Monomial(2, 0, 1, -0.75 * std::sqrt(7 / M_PI)),
              Monomial(0, 2, 1, -0.75 * std::sqrt(7 / M_PI))};
  cs["f+1"] = {Monomial(1, 0, 2, std::sqrt(21. / 2. / M_PI)),
               Monomial(1, 2, 0, -0.25 * std::sqrt(21. / 2. / M_PI)),
               Monomial(3, 0, 0, -0.25 * std::sqrt(21. / 2. / M_PI))};
  cs["f+2"] = {Monomial(2, 0, 1, 0.25 * std::sqrt(105. / M_PI)),
               Monomial(0, 2, 1, -0.25 * std::sqrt(105. / M_PI))};
  cs["f+3"] = {Monomial(3, 0, 0, 0.25 * std::sqrt(35. / 2. / M_PI)),
               Monomial(1, 2, 0, -0.75 * std::sqrt(35. / 2. / M_PI))};

  cs["g-4"] = {Monomial(3, 1, 0, 0.75 * std::sqrt(35. / M_PI)),
               Monomial(1, 3, 0, -0.75 * std::sqrt(35. / M_PI))};
  cs["g-3"] = {Monomial(2, 1, 1, 9.0 * std::sqrt(35. / (2 * M_PI)) / 4.0),
               Monomial(0, 3, 1, -0.75 * std::sqrt(35. / (2. * M_PI)))};
  cs["g-2"] = {Monomial(1, 1, 2, 18.0 * std::sqrt(5. / (M_PI)) / 4.0),
               Monomial(3, 1, 0, -3. * std::sqrt(5. / (M_PI)) / 4.0),
               Monomial(1, 3, 0, -3. * std::sqrt(5. / (M_PI)) / 4.0)};
  cs["g-1"] = {Monomial(0, 1, 3, 3.0 * std::sqrt(5. / (2 * M_PI))),
               Monomial(2, 1, 1, -9.0 * std::sqrt(5. / (2. * M_PI)) / 4.0),
               Monomial(0, 3, 1, -9.0 * std::sqrt(5. / (2. * M_PI)) / 4.0)};
  cs["g0"] = {Monomial(0, 0, 4, 3.0 * std::sqrt(1. / (M_PI)) / 2.0),
              Monomial(4, 0, 0, 9.0 * std::sqrt(1. / (M_PI)) / 16.0),
              Monomial(0, 4, 0, 9.0 * std::sqrt(1. / (M_PI)) / 16.0),
              Monomial(2, 0, 2, -9.0 * std::sqrt(1. / M_PI) / 2.0),
              Monomial(0, 2, 2, -9.0 * std::sqrt(1. / M_PI) / 2.0),
              Monomial(2, 2, 0, 9.0 * std::sqrt(1. / M_PI) / 8.0)};
  cs["g+1"] = {Monomial(1, 0, 3, 3.0 * std::sqrt(5. / (2 * M_PI))),
               Monomial(1, 2, 1, -9.0 * std::sqrt(5. / (2. * M_PI)) / 4.0),
               Monomial(3, 0, 1, -9.0 * std::sqrt(5. / (2. * M_PI)) / 4.0)};
  cs["g+2"] = {Monomial(2, 0, 2, 18.0 * std::sqrt(5. / (M_PI)) / 8.0),
               Monomial(0, 2, 2, -18. * std::sqrt(5. / (M_PI)) / 8.0),
               Monomial(0, 4, 0, 3. * std::sqrt(5. / (M_PI)) / 8.0),
               Monomial(4, 0, 0, -3. * std::sqrt(5. / (M_PI)) / 8.0)};
  cs["g+3"] = {Monomial(1, 2, 1, -9.0 * std::sqrt(35. / (2. * M_PI)) / 4.0),
               Monomial(3, 0, 1, 0.75 * std::sqrt(35. / (2. * M_PI)))};
  cs["g+4"] = {Monomial(4, 0, 0, 3.0 * std::sqrt(35. / M_PI) / 16.0),
               Monomial(2, 2, 0, -18.0 * std::sqrt(35. / M_PI) / 16.0),
               Monomial(0, 4, 0, 3.0 * std::sqrt(35. / M_PI) / 16.0)};

  return cs;
}

struct Basisfunction {
  std::string atom;
  std::array<double, 3> position;
  std::vector<double> alphas;
  std::vector<double> contr_coef;
  std::string lm;

  // Constructor
  Basisfunction(const std::string &atom_,
                const std::array<double, 3> &position_,
                const std::vector<double> &alphas_,
                const std::vector<double> &contr_coef_, const std::string &lm_)
      : atom(atom_), position(position_), alphas(alphas_),
        contr_coef(contr_coef_), lm(lm_) {}
};

// #########################################################################
// ## END Definition of the different Data structs
// #########################################################################

// #########################################################################
// ## Definition of the different function for Overlap computation
// #########################################################################

static double Kcomponent(double Y1k, double Y2k, int ik, int jk, double alpha) {
  double sum = 0.0;

  if (Y1k == 0.0 || Y2k == 0.0) {
    sum = gamma_func(alpha, ik + jk);
  } else {
    for (int o = 0; o <= ik; ++o) {
      for (int p = 0; p <= jk; ++p) {
        if (ik == o && jk == p) {
          sum += gamma_func(alpha, o + p);
        } else {
          sum += gamma_func(alpha, o + p) * binom(ik, o) * binom(jk, p) *
                 std::pow(-Y1k, ik - o) * std::pow(-Y2k, jk - p);
        }
      }
    }
  }

  return sum;
}

static double KFunction(const std::array<double, 3> &Y1,
                        const std::array<double, 3> &Y2,
                        const std::array<int, 3> &iis,
                        const std::array<int, 3> &jjs,
                        double alpha) {

  double output = 1.0;

  for (size_t it = 0; it < Y1.size(); ++it) {
    output *= Kcomponent(Y1[it], Y2[it], iis[it], jjs[it], alpha);
  }

  return output;
}

static std::complex<double> Kqcomponent(double Y1k, double Y2k, int ik, int jk,
                                 double alpha, double qk) {
  std::complex<double> sum(0.0, 0.0);

  if (Y1k == 0.0 || Y2k == 0.0) {
    sum = gq_func(alpha, qk, ik + jk);
  } else {
    for (int o = 0; o <= ik; ++o) {
      for (int p = 0; p <= jk; ++p) {
        if (ik == o && jk == p) {
          sum += gq_func(alpha, qk, o + p);
        } else {
          sum += gq_func(alpha, qk, o + p) * binom(ik, o) * binom(jk, p) *
                 std::pow(-Y1k, ik - o) * std::pow(-Y2k, jk - p);
        }
      }
    }
  }

  return sum;
}

static std::complex<double> KqFunction(const std::array<double, 3> &Y1,
                                const std::array<double, 3> &Y2,
                                const std::array<int, 3> &iis,
                                const std::array<int, 3> &jjs, double alpha,
                                const std::array<double, 3> &q) {

  std::complex<double> output(1.0, 0.0);

  for (size_t it = 0; it < Y1.size(); ++it) {
    output *= Kqcomponent(Y1[it], Y2[it], iis[it], jjs[it], alpha, q[it]);
  }

  return output;
}

static std::complex<double> IJ_q_Int(const std::array<double, 3> &R1,
                              const std::array<double, 3> &R2,
                              const std::vector<Monomial> &lm1,
                              const std::vector<Monomial> &lm2, double A1,
                              double A2, const std::array<double, 3> &q) {
  // computes the IJ integral using the monomial decomposition of the
  // solid harmonics.
  // Input: X of the difference vector R1-R2
  // A1: positive numerical
  // A2: positive numerical

  // Initialize all constants
  std::array<double, 3> X = {R1[0] - R2[0], R1[1] - R2[1], R1[2] - R2[2]};
  std::array<double, 3> Rbar = {A1 * R1[0] / (A1 + A2) + A2 * R2[0] / (A1 + A2),
                                A1 * R1[1] / (A1 + A2) + A2 * R2[1] / (A1 + A2),
                                A1 * R1[2] / (A1 + A2) +
                                    A2 * R2[2] / (A1 + A2)};
  std::complex<double> Jintegral(0.0, 0.0);
  std::array<double, 3> Y1 = {A2 * X[0] / (A1 + A2), A2 * X[1] / (A1 + A2),
                              A2 * X[2] / (A1 + A2)};
  std::array<double, 3> Y2 = {-1.0 * A1 * X[0] / (A1 + A2),
                              -1.0 * A1 * X[1] / (A1 + A2),
                              -1.0 * A1 * X[2] / (A1 + A2)};

  double alpha = A1 + A2;
  // Perform the sum
  for (const auto &Z1 : lm1) {
    for (const auto &Z2 : lm2) {
      Jintegral += Z1.prefactor * Z2.prefactor *
                   KqFunction(Y1, Y2, Z1.indices, Z2.indices, alpha, q);
    }
  }
  double A12red = -1.0 * A1 * A2 / (A1 + A2);
  double Exponent = A12red * (X[0] * X[0] + X[1] * X[1] + X[2] * X[2]);
  double gaussianPrefactor = std::exp(Exponent);
  double dot = q[0] * Rbar[0] + q[1] * Rbar[1] + q[2] * Rbar[2];
  std::complex<double> exponent_q = std::exp(std::complex<double>(0.0, dot));
  std::complex<double> integral = gaussianPrefactor * exponent_q * Jintegral;
  return integral;
}

static double IJ_Int(const std::array<double, 3> &X, const std::vector<Monomial> &lm1,
              const std::vector<Monomial> &lm2, double A1, double A2,
              int modus) {
  // computes the IJ integral using the monomial decomposition of the
  // solid harmonics.
  // Input: X of the difference vector R1-R2
  // A1: positive numerical
  // A2: positive numerical

  // Initialize all constants
  double Jintegral = 0.0;
  std::array<double, 3> Y1 = {A2 * X[0] / (A1 + A2), A2 * X[1] / (A1 + A2),
                              A2 * X[2] / (A1 + A2)};
  std::array<double, 3> Y2 = {-1.0 * A1 * X[0] / (A1 + A2),
                              -1.0 * A1 * X[1] / (A1 + A2),
                              -1.0 * A1 * X[2] / (A1 + A2)};
  double alpha = A1 + A2;
  // Perform the sum
  for (const auto &Z1orig : lm1) {
    auto Z1 = Z1orig; // local, modifiable copy
    if (modus == 0) {
      // no change
    } else if (modus == 1) {
      Z1.increaseIndex1();
    } else if (modus == 2) {
      Z1.increaseIndex2();
    } else if (modus == 3) {
      Z1.increaseIndex3();
    }
    for (const auto &Z2 : lm2) {
      Jintegral += Z1.prefactor * Z2.prefactor *
                   KFunction(Y1, Y2, Z1.indices, Z2.indices, alpha);
    }
  }
  double A12red = -1.0 * A1 * A2 / (A1 + A2);
  double Exponent = A12red * (X[0] * X[0] + X[1] * X[1] + X[2] * X[2]);
  double gaussianPrefactor = std::exp(Exponent);
  double integral = gaussianPrefactor * Jintegral;

  return integral;
}

static double IJ_Int_momentum(const std::array<double, 3> &X,
                       const std::vector<Monomial> &lm1,
                       const std::vector<Monomial> &lm2, double A1, double A2,
                       int modus) {
  // computes the IJ integral using the monomial decomposition of the
  // solid harmonics.
  // Input: X of the difference vector R1-R2
  // A1: positive numerical
  // A2: positive numerical

  // Initialize all constants
  std::array<double, 3> Y1 = {A2 * X[0] / (A1 + A2), A2 * X[1] / (A1 + A2),
                              A2 * X[2] / (A1 + A2)};
  std::array<double, 3> Y2 = {-1.0 * A1 * X[0] / (A1 + A2),
                              -1.0 * A1 * X[1] / (A1 + A2),
                              -1.0 * A1 * X[2] / (A1 + A2)};
  double A12red = -1.0 * A1 * A2 / (A1 + A2);
  double Exponent = A12red * (X[0] * X[0] + X[1] * X[1] + X[2] * X[2]);
  double gaussianPrefactor = std::exp(Exponent);
  double alpha = A1 + A2;

  double first_term = 0.0;
  double n_prefactor = 1.0;
  // Perform the sum
  for (const auto &Z1 : lm1) {
    for (const auto &Z2orig : lm2) {
      auto Z2 = Z2orig;
      if (modus == 1) {
        n_prefactor = Z2.indices[0];
        Z2.decreaseIndex1();
      } else if (modus == 2) {
        n_prefactor = Z2.indices[1];
        Z2.decreaseIndex2();
      } else if (modus == 3) {
        n_prefactor = Z2.indices[2];
        Z2.decreaseIndex3();
      }
      first_term += n_prefactor * Z1.prefactor * Z2.prefactor *
                    KFunction(Y1, Y2, Z1.indices, Z2.indices, alpha);
    }
  }

  double integral_1 = gaussianPrefactor * first_term;

  double second_term = 0.0;
  // Perform the sum
  for (const auto &Z1 : lm1) {
    for (const auto &Z2orig : lm2) {
      auto Z2 = Z2orig;
      if (modus == 1) {
        Z2.increaseIndex1();
      } else if (modus == 2) {
        Z2.increaseIndex2();
      } else if (modus == 3) {
        Z2.increaseIndex3();
      }
      second_term += Z1.prefactor * Z2.prefactor *
                     KFunction(Y1, Y2, Z1.indices, Z2.indices, alpha);
    }
  }

  double integral_2 = (-2) * A2 * gaussianPrefactor * second_term;

  return integral_1 + integral_2;
}

// Function to compute the overlap of two basis functions
static double getoverlap(
    const std::array<double, 3> &R1, const std::vector<double> &contr_coeff1,
    const std::vector<double> &alphas1, const std::vector<Monomial> &lm1,
    const std::array<double, 3> &R2, const std::vector<double> &contr_coeff2,
    const std::vector<double> &alphas2, const std::vector<Monomial> &lm2,
    const std::vector<double> &cell_vectors) {

  double overlap = 0.0;
  // Loop over cell vectors
  for (size_t cell_index = 0; cell_index < cell_vectors.size();
       cell_index += 3) {
    std::array<double, 3> cell_vector = {cell_vectors[cell_index],
                                         cell_vectors[cell_index + 1],
                                         cell_vectors[cell_index + 2]};

    // Compute the shifted positions
    std::array<double, 3> R2_shifted = {
        R2[0] + cell_vector[0], R2[1] + cell_vector[1], R2[2] + cell_vector[2]};

    // Compute the overlap for the shifted positions
    for (size_t it1 = 0; it1 < alphas1.size(); ++it1) {
      for (size_t it2 = 0; it2 < alphas2.size(); ++it2) {
        overlap += contr_coeff1[it1] * contr_coeff2[it2] *
                   IJ_Int({R1[0] - R2_shifted[0], R1[1] - R2_shifted[1],
                           R1[2] - R2_shifted[2]},
                          lm1, lm2, alphas1[it1], alphas2[it2], 0);
      }
    }
  }

  return overlap;
}

// Function to compute the overlap of two basis functions
static double get_r_Matrix_Element(
    const std::array<double, 3> &R1, const std::vector<double> &contr_coeff1,
    const std::vector<double> &alphas1, const std::vector<Monomial> &lm1,
    const std::array<double, 3> &R2, const std::vector<double> &contr_coeff2,
    const std::vector<double> &alphas2, const std::vector<Monomial> &lm2,
    const std::vector<double> &cell_vectors, const int direction) {

  double matrix_element = 0.0;
  // Loop over cell vectors
  for (size_t cell_index = 0; cell_index < cell_vectors.size();
       cell_index += 3) {
    std::array<double, 3> cell_vector = {cell_vectors[cell_index],
                                         cell_vectors[cell_index + 1],
                                         cell_vectors[cell_index + 2]};

    // Compute the shifted positions
    std::array<double, 3> R2_shifted = {
        R2[0] + cell_vector[0], R2[1] + cell_vector[1], R2[2] + cell_vector[2]};

    // Compute the overlap for the shifted positions
    for (size_t it1 = 0; it1 < alphas1.size(); ++it1) {
      for (size_t it2 = 0; it2 < alphas2.size(); ++it2) {
        matrix_element +=
            contr_coeff1[it1] * contr_coeff2[it2] *
                IJ_Int({R1[0] - R2_shifted[0], R1[1] - R2_shifted[1],
                        R1[2] - R2_shifted[2]},
                       lm1, lm2, alphas1[it1], alphas2[it2], direction) +
            contr_coeff1[it1] * contr_coeff2[it2] *
                (R1[direction - 1] - 0.5 * cell_vector[direction - 1]) *
                IJ_Int({R1[0] - R2_shifted[0], R1[1] - R2_shifted[1],
                        R1[2] - R2_shifted[2]},
                       lm1, lm2, alphas1[it1], alphas2[it2], 0);
      }
    }
  }

  return matrix_element;
}

// Function to compute the overlap of two basis functions
static double get_p_Matrix_Element(
    const std::array<double, 3> &R1, const std::vector<double> &contr_coeff1,
    const std::vector<double> &alphas1, const std::vector<Monomial> &lm1,
    const std::array<double, 3> &R2, const std::vector<double> &contr_coeff2,
    const std::vector<double> &alphas2, const std::vector<Monomial> &lm2,
    const std::vector<double> &cell_vectors, const int direction) {

  double matrix_element = 0.0;
  // Loop over cell vectors
  for (size_t cell_index = 0; cell_index < cell_vectors.size();
       cell_index += 3) {
    std::array<double, 3> cell_vector = {cell_vectors[cell_index],
                                         cell_vectors[cell_index + 1],
                                         cell_vectors[cell_index + 2]};

    // Compute the shifted positions
    std::array<double, 3> R2_shifted = {
        R2[0] + cell_vector[0], R2[1] + cell_vector[1], R2[2] + cell_vector[2]};

    // Compute the overlap for the shifted positions
    for (size_t it1 = 0; it1 < alphas1.size(); ++it1) {
      for (size_t it2 = 0; it2 < alphas2.size(); ++it2) {
        matrix_element +=
            contr_coeff1[it1] * contr_coeff2[it2] *
            IJ_Int_momentum({R1[0] - R2_shifted[0], R1[1] - R2_shifted[1],
                             R1[2] - R2_shifted[2]},
                            lm1, lm2, alphas1[it1], alphas2[it2], direction);
      }
    }
  }
  return matrix_element;
}

// Function to compute the overlap of two basis functions
static std::complex<double> get_phase_Matrix_Element(
    const std::array<double, 3> &R1, const std::vector<double> &contr_coeff1,
    const std::vector<double> &alphas1, const std::vector<Monomial> &lm1,
    const std::array<double, 3> &R2, const std::vector<double> &contr_coeff2,
    const std::vector<double> &alphas2, const std::vector<Monomial> &lm2,
    const std::vector<double> &cell_vectors, const std::array<double, 3> &q) {

  std::complex<double> matrix_element(0.0, 0.0);
  // Loop over cell vectors
  for (size_t cell_index = 0; cell_index < cell_vectors.size();
       cell_index += 3) {
    std::array<double, 3> cell_vector = {cell_vectors[cell_index],
                                         cell_vectors[cell_index + 1],
                                         cell_vectors[cell_index + 2]};

    // Compute the shifted positions
    std::array<double, 3> R2_shifted = {
        R2[0] + cell_vector[0], R2[1] + cell_vector[1], R2[2] + cell_vector[2]};

    // Compute the overlap for the shifted positions
    for (size_t it1 = 0; it1 < alphas1.size(); ++it1) {
      for (size_t it2 = 0; it2 < alphas2.size(); ++it2) {
        matrix_element +=
            contr_coeff1[it1] * contr_coeff2[it2] *
            IJ_q_Int(R1, R2_shifted, lm1, lm2, alphas1[it1], alphas2[it2], q);
      }
    }
  }

  return matrix_element;
}

static std::vector<double> get_T_Matrix(
    const std::vector<std::string>& atoms_set1,
    const std::vector<double>& positions_set1,
    const std::vector<double>& alphas_set1,
    const std::vector<int>& alphasLengths_set1,
    const std::vector<double>& contr_coef_set1,
    const std::vector<int>& contr_coefLengths_set1,
    const std::vector<std::string>& lms_set1,
    const std::vector<std::string>& atoms_set2,
    const std::vector<double>& positions_set2,
    const std::vector<double>& alphas_set2,
    const std::vector<int>& alphasLengths_set2,
    const std::vector<double>& contr_coef_set2,
    const std::vector<int>& contr_coefLengths_set2,
    const std::vector<std::string>& lms_set2,
    const std::vector<double>& cell_vectors)
{
    // Initialize the map of monomials
    std::unordered_map<std::string, std::vector<Monomial>> solidHarmonics = getSolidHarmonics();

    // Call by reference for efficiency
    const std::vector<double>& cell_vectors_vec = cell_vectors;

    // --- Construct Basisfunctions for set1 ---
    std::vector<Basisfunction> basisfunctions_set1;
    size_t arrayIndex_set1 = 0, alphasIndex_set1 = 0, contrCoefIndex_set1 = 0;

    for (size_t i = 0; i < atoms_set1.size(); ++i) {
        std::string atom = atoms_set1[i];

        std::array<double, 3> position = {
            positions_set1[arrayIndex_set1],
            positions_set1[arrayIndex_set1 + 1],
            positions_set1[arrayIndex_set1 + 2]
        };
        arrayIndex_set1 += 3;

        std::vector<double> alphas_list(
            alphas_set1.begin() + alphasIndex_set1,
            alphas_set1.begin() + alphasIndex_set1 + alphasLengths_set1[i]
        );
        alphasIndex_set1 += alphasLengths_set1[i];

        std::vector<double> contr_coef_list(
            contr_coef_set1.begin() + contrCoefIndex_set1,
            contr_coef_set1.begin() + contrCoefIndex_set1 + contr_coefLengths_set1[i]
        );
        contrCoefIndex_set1 += contr_coefLengths_set1[i];

        std::string lm = lms_set1[i];

        basisfunctions_set1.emplace_back(atom, position, alphas_list, contr_coef_list, lm);
    }

    // --- Construct Basisfunctions for set2 ---
    std::vector<Basisfunction> basisfunctions_set2;
    size_t arrayIndex_set2 = 0, alphasIndex_set2 = 0, contrCoefIndex_set2 = 0;

    for (size_t i = 0; i < atoms_set2.size(); ++i) {
        std::string atom = atoms_set2[i];

        std::array<double, 3> position = {
            positions_set2[arrayIndex_set2],
            positions_set2[arrayIndex_set2 + 1],
            positions_set2[arrayIndex_set2 + 2]
        };
        arrayIndex_set2 += 3;

        std::vector<double> alphas_list(
            alphas_set2.begin() + alphasIndex_set2,
            alphas_set2.begin() + alphasIndex_set2 + alphasLengths_set2[i]
        );
        alphasIndex_set2 += alphasLengths_set2[i];

        std::vector<double> contr_coef_list(
            contr_coef_set2.begin() + contrCoefIndex_set2,
            contr_coef_set2.begin() + contrCoefIndex_set2 + contr_coefLengths_set2[i]
        );
        contrCoefIndex_set2 += contr_coefLengths_set2[i];

        std::string lm = lms_set2[i];

        basisfunctions_set2.emplace_back(atom, position, alphas_list, contr_coef_list, lm);
    }

    // --- Prepare the output array ---
    std::vector<double> OLPasArray(atoms_set1.size() * atoms_set2.size(), 0.0);

    // Compute overlaps in parallel
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < atoms_set1.size(); ++i) {
        for (size_t j = 0; j < atoms_set2.size(); ++j) {
            double overlap = getoverlap(
                basisfunctions_set1[i].position,
                basisfunctions_set1[i].contr_coef,
                basisfunctions_set1[i].alphas,
                solidHarmonics[basisfunctions_set1[i].lm],
                basisfunctions_set2[j].position,
                basisfunctions_set2[j].contr_coef,
                basisfunctions_set2[j].alphas,
                solidHarmonics[basisfunctions_set2[j].lm],
                cell_vectors_vec
            );

            OLPasArray[i * atoms_set2.size() + j] = overlap;
        }
    }

    return OLPasArray;
}


// Can be deleted after using pybind11 correctly everywhere
static void free_ptr(double *array) {
  // Free the memory allocated for the array
  delete[] array;
}

// Can be deleted after using pybind11 correctly everywhere
static void free_ptr_complex(std::complex<double> *array) {
  // Free the memory allocated for the array
  delete[] array;
}

// #########################################################################
// ## END Definition of the different function for Overlap computation
// #########################################################################

// #########################################################################
// ## Definition of the different function for Real Space Representation
// #########################################################################
static double ExponentialContribution(const std::array<double, 3> &Delta,
                               const std::vector<double> &contr_coeff,
                               const std::vector<double> &alphas) {
  double value = 0.0;
  for (size_t it = 0; it < alphas.size(); ++it) {
    value += contr_coeff[it] *
             std::exp(-alphas[it] * (Delta[0] * Delta[0] + Delta[1] * Delta[1] +
                                     Delta[2] * Delta[2]));
  }
  return value;
}

static double PolynomialContribution(const std::array<double, 3> &Delta,
                              const std::vector<Monomial> &lm) {
  double value = 0.0;
  for (const auto &Z : lm) {
    value += Z.prefactor * pow(Delta[0], Z.indices[0]) *
             pow(Delta[1], Z.indices[1]) * pow(Delta[2], Z.indices[2]);
  }
  return value;
}

// Function to compute the a real space grid representation of a Basisfct.
static double getBasisFunctionOnGrid(const std::array<double, 3> &r,
                              const std::array<double, 3> &R,
                              const std::vector<double> &contr_coeff,
                              const std::vector<double> &alphas,
                              const std::vector<Monomial> &lm,
                              const std::vector<double> &cell_vectors) {
  double WFNvalue = 0.0;
  // Loop over cell vectors
  for (size_t cell_index = 0; cell_index < cell_vectors.size();
       cell_index += 3) {
    std::array<double, 3> cell_vector = {cell_vectors[cell_index],
                                         cell_vectors[cell_index + 1],
                                         cell_vectors[cell_index + 2]};
    std::array<double, 3> Delta = {r[0] - R[0] - cell_vector[0],
                                   r[1] - R[1] - cell_vector[1],
                                   r[2] - R[2] - cell_vector[2]};
    WFNvalue += PolynomialContribution(Delta, lm) *
                ExponentialContribution(Delta, contr_coeff, alphas);
  }
  return WFNvalue;
}

static std::vector<double> get_WFN_On_Grid(
    const std::vector<double>& xyzgrid,
    const std::vector<double>& WFNcoefficients,
    const std::vector<std::string>& atoms_set,
    const std::vector<double>& positions_set,
    const std::vector<double>& alphas_set,
    const std::vector<int>& alphasLengths_set,
    const std::vector<double>& contr_coef_set,
    const std::vector<int>& contr_coefLengths_set,
    const std::vector<std::string>& lms_set,
    const std::vector<double>& cell_vectors)
{
    // Initialize the map of monomials
    std::unordered_map<std::string, std::vector<Monomial>> solidHarmonics = getSolidHarmonics();

    const std::vector<double>& cell_vectors_vec = cell_vectors;

    // Construct Basisfunctions
    std::vector<Basisfunction> basisfunctions_set;
    size_t arrayIndex = 0, alphasIndex = 0, contrCoefIndex = 0;

    for (size_t i = 0; i < atoms_set.size(); ++i) {
        std::string atom = atoms_set[i];

        std::array<double, 3> position = {
            positions_set[arrayIndex],
            positions_set[arrayIndex + 1],
            positions_set[arrayIndex + 2]
        };
        arrayIndex += 3;

        std::vector<double> alphas_list(
            alphas_set.begin() + alphasIndex,
            alphas_set.begin() + alphasIndex + alphasLengths_set[i]
        );
        alphasIndex += alphasLengths_set[i];

        std::vector<double> contr_coef_list(
            contr_coef_set.begin() + contrCoefIndex,
            contr_coef_set.begin() + contrCoefIndex + contr_coefLengths_set[i]
        );
        contrCoefIndex += contr_coefLengths_set[i];

        std::string lm = lms_set[i];

        basisfunctions_set.emplace_back(atom, position, alphas_list, contr_coef_list, lm);
    }

    // Prepare output array
    std::vector<double> WFNonGridArray(xyzgrid.size() / 3);

    // Evaluate wavefunction on grid
#pragma omp parallel for
    for (size_t i = 0; i < xyzgrid.size(); i += 3) {
        std::array<double, 3> r = {xyzgrid[i], xyzgrid[i + 1], xyzgrid[i + 2]};
        double WFNvalue = 0.0;

        for (size_t j = 0; j < atoms_set.size(); ++j) {
            WFNvalue += WFNcoefficients[j] *
                        getBasisFunctionOnGrid(
                            r,
                            basisfunctions_set[j].position,
                            basisfunctions_set[j].contr_coef,
                            basisfunctions_set[j].alphas,
                            solidHarmonics[basisfunctions_set[j].lm],
                            cell_vectors_vec
                        );
        }
        WFNonGridArray[i / 3] = WFNvalue;
    }

    return WFNonGridArray;
}

static std::vector<double> get_Local_Potential_On_Grid(
    const std::vector<double>& xyzgrid,
    const std::vector<double>& MatrixElements,
    const std::vector<std::string>& atoms_set,
    const std::vector<double>& positions_set,
    const std::vector<double>& alphas_set,
    const std::vector<int>& alphasLengths_set,
    const std::vector<double>& contr_coef_set,
    const std::vector<int>& contr_coefLengths_set,
    const std::vector<std::string>& lms_set,
    const std::vector<double>& cell_vectors)
{
    // Initialize the map of monomials
    std::unordered_map<std::string, std::vector<Monomial>> solidHarmonics = getSolidHarmonics();

    // References to input vectors for convenience
    const std::vector<double>& cell_vectors_vec = cell_vectors;
    const std::vector<double>& MatrixElements_vec = MatrixElements;

    // Construct Basisfunctions
    std::vector<Basisfunction> basisfunctions_set;
    size_t arrayIndex = 0, alphasIndex = 0, contrCoefIndex = 0;

    for (size_t i = 0; i < atoms_set.size(); ++i) {
        std::string atom = atoms_set[i];

        std::array<double, 3> position = {
            positions_set[arrayIndex],
            positions_set[arrayIndex + 1],
            positions_set[arrayIndex + 2]
        };
        arrayIndex += 3;

        std::vector<double> alphas_list(
            alphas_set.begin() + alphasIndex,
            alphas_set.begin() + alphasIndex + alphasLengths_set[i]
        );
        alphasIndex += alphasLengths_set[i];

        std::vector<double> contr_coef_list(
            contr_coef_set.begin() + contrCoefIndex,
            contr_coef_set.begin() + contrCoefIndex + contr_coefLengths_set[i]
        );
        contrCoefIndex += contr_coefLengths_set[i];

        std::string lm = lms_set[i];

        basisfunctions_set.emplace_back(atom, position, alphas_list, contr_coef_list, lm);
    }

    // Prepare output array
    std::vector<double> LocalPotentialOnGridArray(xyzgrid.size() / 3);

    // Compute local potential on the grid
#pragma omp parallel for
    for (size_t i = 0; i < xyzgrid.size(); i += 3) {
        std::array<double, 3> r = {xyzgrid[i], xyzgrid[i + 1], xyzgrid[i + 2]};
        double Local_Potential_Value = 0.0;

        for (size_t j = 0; j < atoms_set.size(); ++j) {
            for (size_t k = 0; k < atoms_set.size(); ++k) {
                Local_Potential_Value +=
                    MatrixElements_vec[j * atoms_set.size() + k] *
                    getBasisFunctionOnGrid(
                        r, basisfunctions_set[j].position,
                        basisfunctions_set[j].contr_coef, basisfunctions_set[j].alphas,
                        solidHarmonics[basisfunctions_set[j].lm], cell_vectors_vec
                    ) *
                    getBasisFunctionOnGrid(
                        r, basisfunctions_set[k].position,
                        basisfunctions_set[k].contr_coef, basisfunctions_set[k].alphas,
                        solidHarmonics[basisfunctions_set[k].lm], cell_vectors_vec
                    );
            }
        }
        LocalPotentialOnGridArray[i / 3] = Local_Potential_Value;
    }

    return LocalPotentialOnGridArray;
}

static std::vector<double> get_position_operators(
    const std::vector<std::string>& atoms_set1,
    const std::vector<double>& positions_set1,
    const std::vector<double>& alphas_set1,
    const std::vector<int>& alphasLengths_set1,
    const std::vector<double>& contr_coef_set1,
    const std::vector<int>& contr_coefLengths_set1,
    const std::vector<std::string>& lms_set1,
    const std::vector<std::string>& atoms_set2,
    const std::vector<double>& positions_set2,
    const std::vector<double>& alphas_set2,
    const std::vector<int>& alphasLengths_set2,
    const std::vector<double>& contr_coef_set2,
    const std::vector<int>& contr_coefLengths_set2,
    const std::vector<std::string>& lms_set2,
    const std::vector<double>& cell_vectors,
    int direction)
{
    // Monomials map
    std::unordered_map<std::string, std::vector<Monomial>> solidHarmonics = getSolidHarmonics();
    const std::vector<double>& cell_vectors_vec = cell_vectors;

    // Basisfunctions set 1
    std::vector<Basisfunction> basisfunctions_set1;
    size_t arrayIndex1 = 0, alphasIndex1 = 0, contrCoefIndex1 = 0;

    for (size_t i = 0; i < atoms_set1.size(); ++i) {
        std::array<double, 3> position = {
            positions_set1[arrayIndex1],
            positions_set1[arrayIndex1 + 1],
            positions_set1[arrayIndex1 + 2]
        };
        arrayIndex1 += 3;

        std::vector<double> alphas_list(
            alphas_set1.begin() + alphasIndex1,
            alphas_set1.begin() + alphasIndex1 + alphasLengths_set1[i]
        );
        alphasIndex1 += alphasLengths_set1[i];

        std::vector<double> contr_coef_list(
            contr_coef_set1.begin() + contrCoefIndex1,
            contr_coef_set1.begin() + contrCoefIndex1 + contr_coefLengths_set1[i]
        );
        contrCoefIndex1 += contr_coefLengths_set1[i];

        basisfunctions_set1.emplace_back(atoms_set1[i], position, alphas_list, contr_coef_list, lms_set1[i]);
    }

    // Basisfunctions set 2
    std::vector<Basisfunction> basisfunctions_set2;
    size_t arrayIndex2 = 0, alphasIndex2 = 0, contrCoefIndex2 = 0;

    for (size_t i = 0; i < atoms_set2.size(); ++i) {
        std::array<double, 3> position = {
            positions_set2[arrayIndex2],
            positions_set2[arrayIndex2 + 1],
            positions_set2[arrayIndex2 + 2]
        };
        arrayIndex2 += 3;

        std::vector<double> alphas_list(
            alphas_set2.begin() + alphasIndex2,
            alphas_set2.begin() + alphasIndex2 + alphasLengths_set2[i]
        );
        alphasIndex2 += alphasLengths_set2[i];

        std::vector<double> contr_coef_list(
            contr_coef_set2.begin() + contrCoefIndex2,
            contr_coef_set2.begin() + contrCoefIndex2 + contr_coefLengths_set2[i]
        );
        contrCoefIndex2 += contr_coefLengths_set2[i];

        basisfunctions_set2.emplace_back(atoms_set2[i], position, alphas_list, contr_coef_list, lms_set2[i]);
    }

    // Prepare output array
    std::vector<double> OLPasArray(atoms_set1.size() * atoms_set2.size());

    // Compute overlaps
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < atoms_set1.size(); ++i) {
        for (size_t j = 0; j < atoms_set2.size(); ++j) {
            double overlap = get_r_Matrix_Element(
                basisfunctions_set1[i].position, basisfunctions_set1[i].contr_coef, basisfunctions_set1[i].alphas,
                solidHarmonics[basisfunctions_set1[i].lm],
                basisfunctions_set2[j].position, basisfunctions_set2[j].contr_coef, basisfunctions_set2[j].alphas,
                solidHarmonics[basisfunctions_set2[j].lm], cell_vectors_vec, direction
            );

            OLPasArray[i * atoms_set2.size() + j] = overlap;
        }
    }

    return OLPasArray;
}

static std::vector<double> get_Momentum_Operators(
    const std::vector<std::string>& atoms_set1,
    const std::vector<double>& positions_set1,
    const std::vector<double>& alphas_set1,
    const std::vector<int>& alphasLengths_set1,
    const std::vector<double>& contr_coef_set1,
    const std::vector<int>& contr_coefLengths_set1,
    const std::vector<std::string>& lms_set1,
    const std::vector<std::string>& atoms_set2,
    const std::vector<double>& positions_set2,
    const std::vector<double>& alphas_set2,
    const std::vector<int>& alphasLengths_set2,
    const std::vector<double>& contr_coef_set2,
    const std::vector<int>& contr_coefLengths_set2,
    const std::vector<std::string>& lms_set2,
    const std::vector<double>& cell_vectors,
    int direction)
{
    // Monomials map
    std::unordered_map<std::string, std::vector<Monomial>> solidHarmonics = getSolidHarmonics();
    const std::vector<double>& cell_vectors_vec = cell_vectors;

    // Basisfunctions set 1
    std::vector<Basisfunction> basisfunctions_set1;
    size_t arrayIndex1 = 0, alphasIndex1 = 0, contrCoefIndex1 = 0;
    for (size_t i = 0; i < atoms_set1.size(); ++i) {
        std::array<double, 3> position = {
            positions_set1[arrayIndex1],
            positions_set1[arrayIndex1 + 1],
            positions_set1[arrayIndex1 + 2]
        };
        arrayIndex1 += 3;

        std::vector<double> alphas_list(
            alphas_set1.begin() + alphasIndex1,
            alphas_set1.begin() + alphasIndex1 + alphasLengths_set1[i]
        );
        alphasIndex1 += alphasLengths_set1[i];

        std::vector<double> contr_coef_list(
            contr_coef_set1.begin() + contrCoefIndex1,
            contr_coef_set1.begin() + contrCoefIndex1 + contr_coefLengths_set1[i]
        );
        contrCoefIndex1 += contr_coefLengths_set1[i];

        basisfunctions_set1.emplace_back(atoms_set1[i], position, alphas_list, contr_coef_list, lms_set1[i]);
    }

    // Basisfunctions set 2
    std::vector<Basisfunction> basisfunctions_set2;
    size_t arrayIndex2 = 0, alphasIndex2 = 0, contrCoefIndex2 = 0;
    for (size_t i = 0; i < atoms_set2.size(); ++i) {
        std::array<double, 3> position = {
            positions_set2[arrayIndex2],
            positions_set2[arrayIndex2 + 1],
            positions_set2[arrayIndex2 + 2]
        };
        arrayIndex2 += 3;

        std::vector<double> alphas_list(
            alphas_set2.begin() + alphasIndex2,
            alphas_set2.begin() + alphasIndex2 + alphasLengths_set2[i]
        );
        alphasIndex2 += alphasLengths_set2[i];

        std::vector<double> contr_coef_list(
            contr_coef_set2.begin() + contrCoefIndex2,
            contr_coef_set2.begin() + contrCoefIndex2 + contr_coefLengths_set2[i]
        );
        contrCoefIndex2 += contr_coefLengths_set2[i];

        basisfunctions_set2.emplace_back(atoms_set2[i], position, alphas_list, contr_coef_list, lms_set2[i]);
    }

    // Prepare output array
    std::vector<double> OLPasArray(atoms_set1.size() * atoms_set2.size());

    // Compute momentum operator elements
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < atoms_set1.size(); ++i) {
        for (size_t j = 0; j < atoms_set2.size(); ++j) {
            double momentum_element = get_p_Matrix_Element(
                basisfunctions_set1[i].position, basisfunctions_set1[i].contr_coef, basisfunctions_set1[i].alphas,
                solidHarmonics[basisfunctions_set1[i].lm],
                basisfunctions_set2[j].position, basisfunctions_set2[j].contr_coef, basisfunctions_set2[j].alphas,
                solidHarmonics[basisfunctions_set2[j].lm], cell_vectors_vec, direction
            );

            OLPasArray[i * atoms_set2.size() + j] = momentum_element;
        }
    }

    return OLPasArray;
}


static std::vector<std::complex<double>> get_Phase_Operators(
    const std::vector<std::string>& atoms_set1,
    const std::vector<double>& positions_set1,
    const std::vector<double>& alphas_set1,
    const std::vector<int>& alphasLengths_set1,
    const std::vector<double>& contr_coef_set1,
    const std::vector<int>& contr_coefLengths_set1,
    const std::vector<std::string>& lms_set1,
    const std::vector<std::string>& atoms_set2,
    const std::vector<double>& positions_set2,
    const std::vector<double>& alphas_set2,
    const std::vector<int>& alphasLengths_set2,
    const std::vector<double>& contr_coef_set2,
    const std::vector<int>& contr_coefLengths_set2,
    const std::vector<std::string>& lms_set2,
    const std::vector<double>& cell_vectors,
    const std::vector<double>& q) {

    if (q.size() != 3) {
        throw std::runtime_error("q vector must have exactly 3 elements");
    }
    std::array<double, 3> q_array = { q[0], q[1], q[2] };

    // Map of monomials
    std::unordered_map<std::string, std::vector<Monomial>> solidHarmonics = getSolidHarmonics();

    const std::vector<double>& cell_vectors_vec = cell_vectors;

    // Construct Basisfunction instances for set 1
    std::vector<Basisfunction> basisfunctions_set1;
    int arrayIndex_set1 = 0, alphasIndex_set1 = 0, contrCoefIndex_set1 = 0;

    for (size_t i = 0; i < atoms_set1.size(); ++i) {
        std::string atom = atoms_set1[i];
        std::array<double,3> position = { positions_set1[arrayIndex_set1],
                                          positions_set1[arrayIndex_set1 + 1],
                                          positions_set1[arrayIndex_set1 + 2] };
        int alphasLength = alphasLengths_set1[i];
        std::vector<double> alphas_list_set1(alphas_set1.begin() + alphasIndex_set1,
                                        alphas_set1.begin() + alphasIndex_set1 + alphasLength);
        alphasIndex_set1 += alphasLength;

        int contrCoefLength = contr_coefLengths_set1[i];
        std::vector<double> contr_coef_list_set1(contr_coef_set1.begin() + contrCoefIndex_set1,
                                            contr_coef_set1.begin() + contrCoefIndex_set1 + contrCoefLength);
        contrCoefIndex_set1 += contrCoefLength;

        std::string lm = lms_set1[i];

        basisfunctions_set1.emplace_back(atom, position, alphas_list_set1, contr_coef_list_set1, lm);
        arrayIndex_set1 += 3;
    }

    // Construct Basisfunction instances for set 2
    std::vector<Basisfunction> basisfunctions_set2;
    int arrayIndex_set2 = 0, alphasIndex_set2 = 0, contrCoefIndex_set2 = 0;

    for (size_t i = 0; i < atoms_set2.size(); ++i) {
        std::string atom = atoms_set2[i];
        std::array<double,3> position = { positions_set2[arrayIndex_set2],
                                          positions_set2[arrayIndex_set2 + 1],
                                          positions_set2[arrayIndex_set2 + 2] };
        int alphasLength = alphasLengths_set2[i];
        std::vector<double> alphas_list_set2(alphas_set2.begin() + alphasIndex_set2,
                                        alphas_set2.begin() + alphasIndex_set2 + alphasLength);
        alphasIndex_set2 += alphasLength;

        int contrCoefLength = contr_coefLengths_set2[i];
        std::vector<double> contr_coef_list_set2(contr_coef_set2.begin() + contrCoefIndex_set2,
                                            contr_coef_set2.begin() + contrCoefIndex_set2 + contrCoefLength);
        contrCoefIndex_set2 += contrCoefLength;

        std::string lm = lms_set2[i];

        basisfunctions_set2.emplace_back(atom, position, alphas_list_set2, contr_coef_list_set2, lm);
        arrayIndex_set2 += 3;
    }

    std::vector<std::complex<double>> OLPasArray(atoms_set1.size() * atoms_set2.size());

    // Compute phase operator matrix elements
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < atoms_set1.size(); ++i) {
        for (size_t j = 0; j < atoms_set2.size(); ++j) {
            OLPasArray[i * atoms_set2.size() + j] =
                get_phase_Matrix_Element(
                    basisfunctions_set1[i].position,
                    basisfunctions_set1[i].contr_coef,
                    basisfunctions_set1[i].alphas,
                    solidHarmonics[basisfunctions_set1[i].lm],
                    basisfunctions_set2[j].position,
                    basisfunctions_set2[j].contr_coef,
                    basisfunctions_set2[j].alphas,
                    solidHarmonics[basisfunctions_set2[j].lm],
                    cell_vectors_vec,
                    q_array
                );
        }
    }

    return OLPasArray;
}


// expose to python via pybind11
PYBIND11_MODULE(_extension, m) {
    m.doc() = "Atomic Basis cpp extension module";

    m.def(
        "get_T_Matrix",
        &get_T_Matrix,
        py::arg("atoms_set1"),
        py::arg("positions_set1"),
        py::arg("alphas_set1"),
        py::arg("alphasLengths_set1"),
        py::arg("contr_coef_set1"),
        py::arg("contr_coefLengths_set1"),
        py::arg("lms_set1"),
        py::arg("atoms_set2"),
        py::arg("positions_set2"),
        py::arg("alphas_set2"),
        py::arg("alphasLengths_set2"),
        py::arg("contr_coef_set2"),
        py::arg("contr_coefLengths_set2"),
        py::arg("lms_set2"),
        py::arg("cell_vectors"),
        "Get transformation matrix"
    );
    m.def(
        "get_WFN_On_Grid",
        &get_WFN_On_Grid,
        py::arg("xyzgrid"),
        py::arg("WFNcoefficients"),
        py::arg("atoms_set"),
        py::arg("positions_set"),
        py::arg("alphas_set"),
        py::arg("alphasLengths_set"),
        py::arg("contr_coef_set"),
        py::arg("contr_coefLengths_set"),
        py::arg("lms_set"),
        py::arg("cell_vectors"),
        "Get wavefunction on grid"
    );
    m.def(
        "get_Local_Potential_On_Grid",
        &get_Local_Potential_On_Grid,
        py::arg("xyzgrid"),
        py::arg("MatrixElements"),
        py::arg("atoms_set"),
        py::arg("positions_set"),
        py::arg("alphas_set"),
        py::arg("alphasLengths_set"),
        py::arg("contr_coef_set"),
        py::arg("contr_coefLengths_set"),
        py::arg("lms_set"),
        py::arg("cell_vectors"),
        "Get local potential on grid"
    );
    m.def(
        "get_position_operators",
        &get_position_operators,
        py::arg("atoms_set1"),
        py::arg("positions_set1"),
        py::arg("alphas_set1"),
        py::arg("alphasLengths_set1"),
        py::arg("contr_coef_set1"),
        py::arg("contr_coefLengths_set1"),
        py::arg("lms_set1"),
        py::arg("atoms_set2"),
        py::arg("positions_set2"),
        py::arg("alphas_set2"),
        py::arg("alphasLengths_set2"),
        py::arg("contr_coef_set2"),
        py::arg("contr_coefLengths_set2"),
        py::arg("lms_set2"),
        py::arg("cell_vectors"),
        py::arg("direction"),
        "Get position operator matrix elements"
    );
    m.def(
        "get_Momentum_Operators",
        &get_Momentum_Operators,
        py::arg("atoms_set1"),
        py::arg("positions_set1"),
        py::arg("alphas_set1"),
        py::arg("alphasLengths_set1"),
        py::arg("contr_coef_set1"),
        py::arg("contr_coefLengths_set1"),
        py::arg("lms_set1"),
        py::arg("atoms_set2"),
        py::arg("positions_set2"),
        py::arg("alphas_set2"),
        py::arg("alphasLengths_set2"),
        py::arg("contr_coef_set2"),
        py::arg("contr_coefLengths_set2"),
        py::arg("lms_set2"),
        py::arg("cell_vectors"),
        py::arg("direction"),
        "Get momentum operator matrix elements"
    );
    m.def(
        "get_Phase_Operators",
        &get_Phase_Operators,
        py::arg("atoms_set1"),
        py::arg("positions_set1"),
        py::arg("alphas_set1"),
        py::arg("alphasLengths_set1"),
        py::arg("contr_coef_set1"),
        py::arg("contr_coefLengths_set1"),
        py::arg("lms_set1"),
        py::arg("atoms_set2"),
        py::arg("positions_set2"),
        py::arg("alphas_set2"),
        py::arg("alphasLengths_set2"),
        py::arg("contr_coef_set2"),
        py::arg("contr_coefLengths_set2"),
        py::arg("lms_set2"),
        py::arg("cell_vectors"),
        py::arg("q"),
        "Get phase operator matrix elements"
    );
}

