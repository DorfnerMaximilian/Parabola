#include <iostream>
#include <cmath>
#include <vector>
#include <array>
#include <unordered_map>



//#########################################################################
//## Helper Functions
//#########################################################################


//Implementation of the double factorial function
extern "C" {
double doublefactorial(int n)
{
       if (n == -1 || n == 0) {
        return 1.0;
    } else {
        return n * doublefactorial(n - 2);
    }
}
//Implementation of the binomial function
double binom(int n, int k) {
        double result = 1.0;
        for (int i = 1; i <= k; ++i) {
            result *= (n - i + 1) / static_cast<double>(i);
        }
        return result;
    }

double gamma_func(double alpha,int n)
{
double value=0.0;
if (n%2==0){
    
        value=(doublefactorial(n-1)*sqrt(M_PI))/(pow(2, 0.5 * static_cast<double>(n)) * pow(alpha, 0.5 * static_cast<double>(n) + 0.5));
}
    return value;
}

//#########################################################################
//## END Helper Functions
//#########################################################################

//#########################################################################
//## Definition of the different Data structs
//#########################################################################

struct Monomial {
    std::array<int,3> indices;
    double prefactor;
    // Constructor
    Monomial(int index1, int index2, int index3, double factor) 
        : indices{index1, index2, index3}, prefactor(factor) {}
    // Method to increase index1 by 1
    void increaseIndex1() {
        indices[0]++;
    }

    // Method to decrease index1 by 1
    void decreaseIndex1() {
        indices[0]--;
    }

    // Method to increase index2 by 1
    void increaseIndex2() {
        indices[1]++;
    }

    // Method to decrease index2 by 1
    void decreaseIndex2() {
        indices[1]--;
    }

    // Method to increase index3 by 1
    void increaseIndex3() {
        indices[2]++;
    }

    // Method to decrease index3 by 1
    void decreaseIndex3() {
        indices[2]--;
    }
};


 std::unordered_map<std::string, std::vector<Monomial> > getSolidHarmonics(){
std::unordered_map<std::string, std::vector<Monomial> > cs;
cs["s"] = {Monomial(0, 0, 0, 0.5*std::sqrt(1.0/ M_PI))};

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
cs["g0"] = {Monomial(0, 0, 4, 3.0 * std::sqrt(1. / (M_PI)) / 2.0),Monomial(4, 0, 0, 9.0 * std::sqrt(1. / (M_PI)) / 16.0),Monomial(0, 4, 0, 9.0 * std::sqrt(1. / (M_PI)) / 16.0),Monomial(2, 0, 2, -9.0 * std::sqrt(1. / M_PI) / 2.0),Monomial(0, 2, 2, -9.0 * std::sqrt(1. / M_PI) / 2.0),Monomial(2, 2, 0, 9.0 * std::sqrt(1. / M_PI) / 8.0)};
cs["g+1"] = {Monomial(1, 0, 3, 3.0 * std::sqrt(5. / (2 * M_PI))),Monomial(1, 2, 1, -9.0 * std::sqrt(5. / (2. * M_PI)) / 4.0),Monomial(3, 0, 1, -9.0 * std::sqrt(5. / (2. * M_PI)) / 4.0)};
cs["g+2"] = {Monomial(2, 0, 2, 18.0 * std::sqrt(5. / (M_PI)) / 8.0),Monomial(0, 2, 2, -18. * std::sqrt(5. / (M_PI)) / 8.0),Monomial(0, 4, 0, 3. * std::sqrt(5. / (M_PI)) / 8.0),Monomial(4, 0, 0, -3. * std::sqrt(5. / (M_PI)) / 8.0)};
cs["g+3"] = {Monomial(1, 2, 1, -9.0 * std::sqrt(35. / (2. * M_PI)) / 4.0),Monomial(3, 0, 1, 0.75 * std::sqrt(35. / (2. * M_PI)))};
cs["g+4"] = {Monomial(4, 0, 0, 3.0 * std::sqrt(35. / M_PI) / 16.0),Monomial(2, 2, 0, -18.0 * std::sqrt(35. / M_PI) / 16.0),Monomial(0, 4, 0, 3.0 * std::sqrt(35. / M_PI) / 16.0)};

return cs;
 }

struct Basisfunction {
    std::string atom;
    std::array<double, 3> position;
    std::vector<double> alphas;
    std::vector<double> contr_coef;
    std::string lm;

    // Constructor
    Basisfunction(const std::string& atom_,
                  const std::array<double, 3>& position_,
                  const std::vector<double>& alphas_,
                  const std::vector<double>& contr_coef_,
                  const std::string& lm_)
        : atom(atom_), position(position_), alphas(alphas_), contr_coef(contr_coef_), lm(lm_) {}
};


//#########################################################################
//## END Definition of the different Data structs
//#########################################################################



//#########################################################################
//## Definition of the different function for Overlap computation
//#########################################################################

double Kcomponent(double Y1k, double Y2k, int ik, int jk, double alpha){
    double sum = 0.0;

    if (Y1k == 0.0 || Y2k == 0.0) {
        sum = gamma_func(alpha, ik + jk);
    } else {
        for (int o = 0; o <= ik; ++o) {
            for (int p = 0; p <= jk; ++p) {
                if (ik == o && jk == p) {
                    sum += gamma_func(alpha, o + p);
                } else {
                    sum += gamma_func(alpha, o + p) * binom(ik, o) * binom(jk, p) * pow(-Y1k, ik - o) * pow(-Y2k, jk - p);
                }
            }
        }
    }

    return sum;
}
double KFunction(const std::array<double,3>& Y1, const std::array<double,3>& Y2,
                 const std::array<int,3>& iis, const std::array<int,3>& jjs, double alpha) {

    double output = 1.0;

    for (size_t it = 0; it < Y1.size(); ++it) {
        output *= Kcomponent(Y1[it], Y2[it], iis[it], jjs[it],alpha);
    }

    return output;
}


double IJ_Int(const std::array<double, 3>& X, const std::vector<Monomial>& lm1,const std::vector<Monomial>& lm2, double A1, double A2,int modus) {
    // computes the IJ integral using the monomial decomposition of the 
    // solid harmonics.
    //Input: X of the difference vector R1-R2
    //A1: positive numerical
    //A2: positive numerical
    
    //Initialize all constants
    double Jintegral=0.0;
    std::array<double,3> Y1={A2*X[0]/(A1+A2),A2*X[1]/(A1+A2),A2*X[2]/(A1+A2)};
    std::array<double,3> Y2={-1.0*A1*X[0]/(A1+A2),-1.0*A1*X[1]/(A1+A2),-1.0*A1*X[2]/(A1+A2)};
    double alpha=A1+A2;
    //Perform the sum 
    for (const auto& Z1orig : lm1) {
        auto Z1 = Z1orig;  // local, modifiable copy
        if (modus == 0) {
            // no change
        } else if (modus == 1) {
            Z1.increaseIndex1();
        } else if (modus == 2) {
            Z1.increaseIndex2();
        } else if (modus == 3) {
            Z1.increaseIndex3();
        }
        for (const auto& Z2 : lm2) {
            Jintegral += Z1.prefactor * Z2.prefactor * KFunction(Y1, Y2, Z1.indices, Z2.indices, alpha);
        }
    }
    double A12red = -1.0*A1 * A2 / (A1 + A2);
    double Exponent = A12red * (X[0]*X[0]+X[1]*X[1]+X[2]*X[2]);
    double gaussianPrefactor = std::exp(Exponent);
    double integral = gaussianPrefactor * Jintegral;

    return integral;
}

// Function to compute the overlap of two basis functions
double getoverlap(const std::array<double,3>& R1,
                  const std::vector<double>& contr_coeff1,
                  const std::vector<double>& alphas1,
                  const std::vector<Monomial>& lm1,
                  const std::array<double,3>& R2,
                  const std::vector<double>& contr_coeff2,
                  const std::vector<double>& alphas2,
                  const std::vector<Monomial>& lm2,
                  const std::vector<double>& cell_vectors
                ) {

    double overlap = 0.0;
    // Loop over cell vectors
    for (size_t cell_index = 0; cell_index < cell_vectors.size(); cell_index += 3) {
        std::array<double, 3> cell_vector = {cell_vectors[cell_index],
                                            cell_vectors[cell_index + 1],
                                            cell_vectors[cell_index + 2]};

        // Compute the shifted positions
        std::array<double, 3> R2_shifted = {R2[0] + cell_vector[0],
                                           R2[1] + cell_vector[1],
                                           R2[2] + cell_vector[2]};

        // Compute the overlap for the shifted positions
        for (size_t it1 = 0; it1 < alphas1.size(); ++it1) {
            for (size_t it2 = 0; it2 < alphas2.size(); ++it2) {
                overlap += contr_coeff1[it1] * contr_coeff2[it2] *
                           IJ_Int({R1[0] - R2_shifted[0], R1[1] - R2_shifted[1], R1[2] - R2_shifted[2]},
                                lm1, lm2, alphas1[it1], alphas2[it2],0);
            }
        }
    }

    return overlap;
}
// Function to compute the overlap of two basis functions
double get_r_Matrix_Element(const std::array<double,3>& R1,
    const std::vector<double>& contr_coeff1,
    const std::vector<double>& alphas1,
    const std::vector<Monomial>& lm1,
    const std::array<double,3>& R2,
    const std::vector<double>& contr_coeff2,
    const std::vector<double>& alphas2,
    const std::vector<Monomial>& lm2,
    const std::vector<double>& cell_vectors,
    const int direction
  ) {

double matrix_element = 0.0;
// Loop over cell vectors
for (size_t cell_index = 0; cell_index < cell_vectors.size(); cell_index += 3) {
std::array<double, 3> cell_vector = {cell_vectors[cell_index],
                              cell_vectors[cell_index + 1],
                              cell_vectors[cell_index + 2]};

// Compute the shifted positions
std::array<double, 3> R2_shifted = {R2[0] + cell_vector[0],
                             R2[1] + cell_vector[1],
                             R2[2] + cell_vector[2]};

// Compute the overlap for the shifted positions
for (size_t it1 = 0; it1 < alphas1.size(); ++it1) {
for (size_t it2 = 0; it2 < alphas2.size(); ++it2) {
  matrix_element += contr_coeff1[it1] * contr_coeff2[it2] *
             IJ_Int({R1[0] - R2_shifted[0], R1[1] - R2_shifted[1], R1[2] - R2_shifted[2]},
                  lm1, lm2, alphas1[it1], alphas2[it2],direction)+contr_coeff1[it1] * contr_coeff2[it2] *(R1[direction-1]-0.5*cell_vector[direction-1])*IJ_Int({R1[0] - R2_shifted[0], R1[1] - R2_shifted[1], R1[2] - R2_shifted[2]},
                    lm1, lm2, alphas1[it1], alphas2[it2],0);
}
}
}

return matrix_element;
}


double* get_T_Matrix(const char* atoms_set1[],
                                const double positions_set1[],
                                const double alphas_set1[],
                                const int alphasLengths_set1[],
                                const double contr_coef_set1[],
                                const int contr_coefLengths_set1[],
                                const char* lms_set1[],
                                int size_set1,
                                const char* atoms_set2[],
                                const double positions_set2[],
                                const double alphas_set2[],
                                const int alphasLengths_set2[],
                                const double contr_coef_set2[],
                                const int contr_coefLengths_set2[],
                                const char* lms_set2[],
                                int size_set2,
                                const double cell_vectors[],
                                int size_cell_vectors
                                ) {

    // Initialize the map of monomials
    std::unordered_map<std::string, std::vector<Monomial>> solidHarmonics = getSolidHarmonics();
    //Process the cell_vectors
    std::vector<double> cell_vectors_vec(cell_vectors, cell_vectors + size_cell_vectors);


    // Process the input arrays - construct Basisfunction instances for set 1
    std::vector<Basisfunction> basisfunctions_set1;
    int arrayIndex_set1 = 0;
    int alphasIndex_set1 = 0;
    int contrCoefIndex_set1 = 0;

    for (int i = 0; i < size_set1; ++i) {
        std::string atom(atoms_set1[i]);
        std::array<double, 3> position = {positions_set1[arrayIndex_set1],
                                          positions_set1[arrayIndex_set1 + 1],
                                          positions_set1[arrayIndex_set1 + 2]};

        // Extract alphas for the current basis function in set 1
        int alphasLength_set1 = alphasLengths_set1[i];
        std::vector<double> alphas_list_set1(alphas_set1 + alphasIndex_set1, alphas_set1 + alphasIndex_set1 + alphasLength_set1);
        alphasIndex_set1 += alphasLength_set1;

        // Extract contr_coef for the current basis function in set 1
        int contrCoefLength_set1 = contr_coefLengths_set1[i];
        std::vector<double> contr_coef_list_set1(contr_coef_set1 + contrCoefIndex_set1, contr_coef_set1 + contrCoefIndex_set1 + contrCoefLength_set1);
        contrCoefIndex_set1 += contrCoefLength_set1;

        std::string lm(lms_set1[i]);

        Basisfunction basisfunction(atom, position, alphas_list_set1, contr_coef_list_set1, lm);
        basisfunctions_set1.push_back(basisfunction);

        arrayIndex_set1 += 3;  // Increment by 3 for positions (3 elements per position)
    }

    // Process the input arrays - construct Basisfunction instances for set 2
    std::vector<Basisfunction> basisfunctions_set2;
    int arrayIndex_set2 = 0;
    int alphasIndex_set2 = 0;
    int contrCoefIndex_set2 = 0;

    for (int i = 0; i < size_set2; ++i) {
        std::string atom(atoms_set2[i]);
        std::array<double, 3> position = {positions_set2[arrayIndex_set2],
                                          positions_set2[arrayIndex_set2 + 1],
                                          positions_set2[arrayIndex_set2 + 2]};

        // Extract alphas for the current basis function in set 2
        int alphasLength_set2 = alphasLengths_set2[i];
        std::vector<double> alphas_list_set2(alphas_set2 + alphasIndex_set2, alphas_set2 + alphasIndex_set2 + alphasLength_set2);
        alphasIndex_set2 += alphasLength_set2;

        // Extract contr_coef for the current basis function in set 2
        int contrCoefLength_set2 = contr_coefLengths_set2[i];
        std::vector<double> contr_coef_list_set2(contr_coef_set2 + contrCoefIndex_set2, contr_coef_set2 + contrCoefIndex_set2 + contrCoefLength_set2);
        contrCoefIndex_set2 += contrCoefLength_set2;

        std::string lm(lms_set2[i]);

        Basisfunction basisfunction(atom, position, alphas_list_set2, contr_coef_list_set2, lm);
        basisfunctions_set2.push_back(basisfunction);

        arrayIndex_set2 += 3;  // Increment by 3 for positions (3 elements per position)
    }
    double* OLPasArray_ptr = new double[size_set1*size_set2];
    // Now, loop over both sets and compute overlaps
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < size_set1; ++i) {
        for (int j = 0; j < size_set2; ++j) {
            double overlap = getoverlap(basisfunctions_set1[i].position,
                                        basisfunctions_set1[i].contr_coef,
                                        basisfunctions_set1[i].alphas,
                                        solidHarmonics[basisfunctions_set1[i].lm],
                                        basisfunctions_set2[j].position,
                                        basisfunctions_set2[j].contr_coef,
                                        basisfunctions_set2[j].alphas,
                                        solidHarmonics[basisfunctions_set2[j].lm],
                                        cell_vectors_vec
                                    );

            // Assign result to the appropriate index in the output array
            OLPasArray_ptr[i * size_set2 + j] = overlap;
        }
    }
    return OLPasArray_ptr;
}
void free_ptr(double* array) {
    // Free the memory allocated for the array
    delete[] array;
}
//#########################################################################
//## END Definition of the different function for Overlap computation
//#########################################################################

//#########################################################################
//## Definition of the different function for Real Space Representation
//#########################################################################
double ExponentialContribution(const std::array<double,3>&  Delta,
                    const std::vector<double>& contr_coeff,
                    const std::vector<double>& alphas){
                    double value=0.0;
                    for (size_t it = 0; it < alphas.size(); ++it) {
                value += contr_coeff[it] * std::exp(-alphas[it]*(Delta[0]*Delta[0]+Delta[1]*Delta[1]+Delta[2]*Delta[2]));
        }
return value;
                    }
double PolynomialContribution(const std::array<double,3>&  Delta,
                              const std::vector<Monomial>& lm){
    double value=0.0;
    for (const auto& Z : lm) {
        value += Z.prefactor * pow(Delta[0],Z.indices[0])*pow(Delta[1],Z.indices[1])*pow(Delta[2],Z.indices[2]);
        }
return value;
}                 
// Function to compute the a real space grid representation of a Basisfct.
double getBasisFunctionOnGrid(const std::array<double,3>&  r,
                    const std::array<double,3>& R,
                    const std::vector<double>& contr_coeff,
                    const std::vector<double>& alphas,
                    const std::vector<Monomial>& lm,
                    const std::vector<double>& cell_vectors) {
    double WFNvalue = 0.0;
    // Loop over cell vectors
    for (size_t cell_index = 0; cell_index < cell_vectors.size(); cell_index += 3) {
        std::array<double, 3> cell_vector = {cell_vectors[cell_index],cell_vectors[cell_index + 1],cell_vectors[cell_index + 2]};
        std::array<double, 3> Delta= {r[0]-R[0]-cell_vector[0],r[1]-R[1]-cell_vector[1],r[2]-R[2]-cell_vector[2]};
        WFNvalue += PolynomialContribution(Delta,lm)*ExponentialContribution(Delta,contr_coeff,alphas);
        }
    return WFNvalue;
}
double* get_WFN_On_Grid(const double xyzgrid[],
                        int size_xyzgrid,
                        const double WFNcoefficients[],
                        const char* atoms_set[],
                        const double positions_set[],
                        const double alphas_set[],
                        const int alphasLengths_set[],
                        const double contr_coef_set[],
                        const int contr_coefLengths_set[],
                        const char* lms_set[],
                        int size_set,
                        const double cell_vectors[],
                        int size_cell_vectors) {

    // Initialize the map of monomials
    std::unordered_map<std::string, std::vector<Monomial>> solidHarmonics = getSolidHarmonics();
    //Process the cell_vectors
    std::vector<double> cell_vectors_vec(cell_vectors, cell_vectors + size_cell_vectors);
    // Process the xyzgrid
    std::vector<double> xyzgrid_vec(xyzgrid, xyzgrid + size_xyzgrid);
    // Process the WFNcoefficients
    std::vector<double> WFNcoefficients_vec(WFNcoefficients, WFNcoefficients + size_set);
    // Process the input arrays - construct Basisfunction instances
    std::vector<Basisfunction> basisfunctions_set;
    int arrayIndex_set = 0;
    int alphasIndex_set = 0;
    int contrCoefIndex_set = 0;

    for (int i = 0; i < size_set; ++i) {
        std::string atom(atoms_set[i]);
        std::array<double, 3> position = {positions_set[arrayIndex_set],
                                          positions_set[arrayIndex_set + 1],
                                          positions_set[arrayIndex_set + 2]};

        // Extract alphas for the current basis function in set 1
        int alphasLength_set = alphasLengths_set[i];
        std::vector<double> alphas_list_set(alphas_set + alphasIndex_set, alphas_set + alphasIndex_set + alphasLength_set);
        alphasIndex_set += alphasLength_set;

        // Extract contr_coef for the current basis function in set 1
        int contrCoefLength_set = contr_coefLengths_set[i];
        std::vector<double> contr_coef_list_set(contr_coef_set + contrCoefIndex_set, contr_coef_set + contrCoefIndex_set + contrCoefLength_set);
        contrCoefIndex_set += contrCoefLength_set;

        std::string lm(lms_set[i]);

        Basisfunction basisfunction(atom, position, alphas_list_set, contr_coef_list_set, lm);
        basisfunctions_set.push_back(basisfunction);

        arrayIndex_set += 3;  // Increment by 3 for positions (3 elements per position)
    }

    double* WFNonGridArray_ptr = new double[size_xyzgrid/3];
    std::cout<<" size_xyzgrid= "<<size_xyzgrid/3<<"\n";
    // Now, loop 
    #pragma omp parallel for
    for (int i = 0; i < size_xyzgrid; i+=3) {
        double WFNvalue=0.0;
        const std::array<double,3>&  r= {xyzgrid_vec[i],xyzgrid_vec[i+1],xyzgrid_vec[i+2]};
        for (int j = 0; j < size_set; ++j) {
            WFNvalue += WFNcoefficients_vec[j]*getBasisFunctionOnGrid(r,
                                        basisfunctions_set[j].position,
                                        basisfunctions_set[j].contr_coef,
                                        basisfunctions_set[j].alphas,
                                        solidHarmonics[basisfunctions_set[j].lm],
                                        cell_vectors_vec);

            // Assign result to the appropriate index in the output array
        }
        WFNonGridArray_ptr[i/3] = WFNvalue;
    }
    return WFNonGridArray_ptr;
}
double* get_Local_Potential_On_Grid(const double xyzgrid[],
                        int size_xyzgrid,
                        const double MatrixElements[],
                        const char* atoms_set[],
                        const double positions_set[],
                        const double alphas_set[],
                        const int alphasLengths_set[],
                        const double contr_coef_set[],
                        const int contr_coefLengths_set[],
                        const char* lms_set[],
                        int size_set,
                        const double cell_vectors[],
                        int size_cell_vectors) {

    // Initialize the map of monomials
    std::unordered_map<std::string, std::vector<Monomial>> solidHarmonics = getSolidHarmonics();
    //Process the cell_vectors
    std::vector<double> cell_vectors_vec(cell_vectors, cell_vectors + size_cell_vectors);
    // Process the xyzgrid
    std::vector<double> xyzgrid_vec(xyzgrid, xyzgrid + size_xyzgrid);
    // Process the WFNcoefficients
    std::vector<double> MatrixElements_vec(MatrixElements, MatrixElements + size_set*size_set);
    // Process the input arrays - construct Basisfunction instances
    std::vector<Basisfunction> basisfunctions_set;
    int arrayIndex_set = 0;
    int alphasIndex_set = 0;
    int contrCoefIndex_set = 0;

    for (int i = 0; i < size_set; ++i) {
        std::string atom(atoms_set[i]);
        std::array<double, 3> position = {positions_set[arrayIndex_set],
                                          positions_set[arrayIndex_set + 1],
                                          positions_set[arrayIndex_set + 2]};

        // Extract alphas for the current basis function in set 1
        int alphasLength_set = alphasLengths_set[i];
        std::vector<double> alphas_list_set(alphas_set + alphasIndex_set, alphas_set + alphasIndex_set + alphasLength_set);
        alphasIndex_set += alphasLength_set;

        // Extract contr_coef for the current basis function in set 1
        int contrCoefLength_set = contr_coefLengths_set[i];
        std::vector<double> contr_coef_list_set(contr_coef_set + contrCoefIndex_set, contr_coef_set + contrCoefIndex_set + contrCoefLength_set);
        contrCoefIndex_set += contrCoefLength_set;

        std::string lm(lms_set[i]);

        Basisfunction basisfunction(atom, position, alphas_list_set, contr_coef_list_set, lm);
        basisfunctions_set.push_back(basisfunction);

        arrayIndex_set += 3;  // Increment by 3 for positions (3 elements per position)
    }

    double* LocalPotentialOnGridArray_ptr = new double[size_xyzgrid/3];
    std::cout<<" size_xyzgrid= "<<size_xyzgrid/3<<"\n";
    // Now, loop
    #pragma omp parallel for 
    for (int i = 0; i < size_xyzgrid; i+=3) {
        double Local_Potential_Value=0.0;
        const std::array<double,3>&  r= {xyzgrid_vec[i],xyzgrid_vec[i+1],xyzgrid_vec[i+2]};
        for (int j = 0; j < size_set; ++j) {
            for (int k =0;k<size_set;++k){
                Local_Potential_Value += MatrixElements_vec[j * size_set + k]*getBasisFunctionOnGrid(r,basisfunctions_set[j].position,basisfunctions_set[j].contr_coef,basisfunctions_set[j].alphas,solidHarmonics[basisfunctions_set[j].lm],cell_vectors_vec)*getBasisFunctionOnGrid(r,basisfunctions_set[k].position,basisfunctions_set[k].contr_coef,basisfunctions_set[k].alphas,solidHarmonics[basisfunctions_set[k].lm],cell_vectors_vec);
            }
            // Assign result to the appropriate index in the output array
        }
        LocalPotentialOnGridArray_ptr[i/3] = Local_Potential_Value;
    }
    return LocalPotentialOnGridArray_ptr;
}
double* get_Position_Operators(const char* atoms_set1[],
    const double positions_set1[],
    const double alphas_set1[],
    const int alphasLengths_set1[],
    const double contr_coef_set1[],
    const int contr_coefLengths_set1[],
    const char* lms_set1[],
    int size_set1,
    const char* atoms_set2[],
    const double positions_set2[],
    const double alphas_set2[],
    const int alphasLengths_set2[],
    const double contr_coef_set2[],
    const int contr_coefLengths_set2[],
    const char* lms_set2[],
    int size_set2,
    const double cell_vectors[],
    int size_cell_vectors,
    int direction
    ) {

// Initialize the map of monomials
std::unordered_map<std::string, std::vector<Monomial>> solidHarmonics = getSolidHarmonics();
//Process the cell_vectors
std::vector<double> cell_vectors_vec(cell_vectors, cell_vectors + size_cell_vectors);


// Process the input arrays - construct Basisfunction instances for set 1
std::vector<Basisfunction> basisfunctions_set1;
int arrayIndex_set1 = 0;
int alphasIndex_set1 = 0;
int contrCoefIndex_set1 = 0;

for (int i = 0; i < size_set1; ++i) {
std::string atom(atoms_set1[i]);
std::array<double, 3> position = {positions_set1[arrayIndex_set1],
              positions_set1[arrayIndex_set1 + 1],
              positions_set1[arrayIndex_set1 + 2]};

// Extract alphas for the current basis function in set 1
int alphasLength_set1 = alphasLengths_set1[i];
std::vector<double> alphas_list_set1(alphas_set1 + alphasIndex_set1, alphas_set1 + alphasIndex_set1 + alphasLength_set1);
alphasIndex_set1 += alphasLength_set1;

// Extract contr_coef for the current basis function in set 1
int contrCoefLength_set1 = contr_coefLengths_set1[i];
std::vector<double> contr_coef_list_set1(contr_coef_set1 + contrCoefIndex_set1, contr_coef_set1 + contrCoefIndex_set1 + contrCoefLength_set1);
contrCoefIndex_set1 += contrCoefLength_set1;

std::string lm(lms_set1[i]);

Basisfunction basisfunction(atom, position, alphas_list_set1, contr_coef_list_set1, lm);
basisfunctions_set1.push_back(basisfunction);

arrayIndex_set1 += 3;  // Increment by 3 for positions (3 elements per position)
}

// Process the input arrays - construct Basisfunction instances for set 2
std::vector<Basisfunction> basisfunctions_set2;
int arrayIndex_set2 = 0;
int alphasIndex_set2 = 0;
int contrCoefIndex_set2 = 0;

for (int i = 0; i < size_set2; ++i) {
std::string atom(atoms_set2[i]);
std::array<double, 3> position = {positions_set2[arrayIndex_set2],
              positions_set2[arrayIndex_set2 + 1],
              positions_set2[arrayIndex_set2 + 2]};

// Extract alphas for the current basis function in set 2
int alphasLength_set2 = alphasLengths_set2[i];
std::vector<double> alphas_list_set2(alphas_set2 + alphasIndex_set2, alphas_set2 + alphasIndex_set2 + alphasLength_set2);
alphasIndex_set2 += alphasLength_set2;

// Extract contr_coef for the current basis function in set 2
int contrCoefLength_set2 = contr_coefLengths_set2[i];
std::vector<double> contr_coef_list_set2(contr_coef_set2 + contrCoefIndex_set2, contr_coef_set2 + contrCoefIndex_set2 + contrCoefLength_set2);
contrCoefIndex_set2 += contrCoefLength_set2;

std::string lm(lms_set2[i]);

Basisfunction basisfunction(atom, position, alphas_list_set2, contr_coef_list_set2, lm);
basisfunctions_set2.push_back(basisfunction);

arrayIndex_set2 += 3;  // Increment by 3 for positions (3 elements per position)
}
double* OLPasArray_ptr = new double[size_set1*size_set2];
// Now, loop over both sets and compute overlaps
#pragma omp parallel for collapse(2)
for (int i = 0; i < size_set1; ++i) {
for (int j = 0; j < size_set2; ++j) {
double overlap = get_r_Matrix_Element(basisfunctions_set1[i].position,
            basisfunctions_set1[i].contr_coef,
            basisfunctions_set1[i].alphas,
            solidHarmonics[basisfunctions_set1[i].lm],
            basisfunctions_set2[j].position,
            basisfunctions_set2[j].contr_coef,
            basisfunctions_set2[j].alphas,
            solidHarmonics[basisfunctions_set2[j].lm],
            cell_vectors_vec,
            direction
        );

// Assign result to the appropriate index in the output array
OLPasArray_ptr[i * size_set2 + j] = overlap;
}
}
return OLPasArray_ptr;
}

}
