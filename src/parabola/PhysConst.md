# Module Documentation: PhysicalConstantsModule

## Overview:

This module provides dictionaries and functions related to physical constants and other information. The module is designed to be used in scientific computing and simulations where accurate physical constants and conversion factors are required.

## Usage:

1. **StandardAtomicWeights:**
   - Function: `StandardAtomicWeights()`
   - Description: Retrieves a dictionary of standard atomic weights for various elements. The data is sourced from NIST (National Institute of Standards and Technology).
   - Output: Returns a dictionary where keys are atomic symbols (e.g., "H", "He") and values are the corresponding standard atomic weights.

2. **PhysicalConstants:**
   - Function: `PhysicalConstants()`
   - Description: Provides a dictionary containing various physical constants in SI units. The values are sourced from NIST Constants website.
   - Output: Returns a dictionary with keys representing physical constants (e.g., Avogadro constant, Boltzmann constant) and values representing their numerical values in SI units.

3. **ConversionFactors:**
   - Function: `ConversionFactors()`
   - Description: Returns a dictionary containing conversion factors for mass, length, energy, force, and specific coupling constants.
   - Output: Returns a dictionary with keys representing conversion factors (e.g., 'u->a.u.', 'A->a.u.') and values representing the conversion factor between the specified units.

4. **AtomSymbolToAtomnumber:**
   - Function: `AtomSymbolToAtomnumber(Symbol)`
   - Description: Converts an atomic symbol (e.g., "H", "He") to the corresponding atom number.
   - Input: Takes an atomic symbol as a string (e.g., "H", "He").
   - Output: Returns the corresponding atom number as an integer. If the input symbol is not implemented yet, it prints a message and returns -1.

## Data Sources:

- **Standard Atomic Weights:**
  - Data Source: [NIST - Atomic Weights and Isotopic Compositions](https://www.nist.gov/pml/atomic-weights-and-isotopic-compositions-relative-atomic-masses) (Accessed: 10.11.2022)

- **Physical Constants:**
  - Data Source: [NIST - CODATA Fundamental Physical Constants](https://physics.nist.gov/cuu/Constants) (Accessed: 10.11.2022)

## File Dependencies:

- The module assumes the presence of a file named `periodictable.dat` in the same directory, containing data related to the periodic table.

## Example Usage:

```python
# Import the module
from PhysicalConstantsModule import *

# Get standard atomic weights
standard_atomic_weights = StandardAtomicWeights()
print("Standard Atomic Weights:", standard_atomic_weights)

# Get physical constants
physical_constants = PhysicalConstants()
print("Physical Constants:", physical_constants)

# Get conversion factors
conversion_factors = ConversionFactors()
print("Conversion Factors:", conversion_factors)

# Convert atomic symbol to atom number
atom_number = AtomSymbolToAtomnumber("O")
print("Atom Number for Oxygen:", atom_number)
```

Make sure the `periodictable.dat` file is available in the module's directory for accurate operation.
