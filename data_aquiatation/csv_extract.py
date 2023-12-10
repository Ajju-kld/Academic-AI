import pandas as pd

def list_of_dict_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

def csv_to_list_of_dict(filename):
    df = pd.read_csv(filename)
    return df.to_dict('records')

# Example usage
data = [
   # Module 1: Linear Algebra
    {'subject': 'Linear Algebra and Calculus', 'description': 'Systems of Linear Equations', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': None, 'deadline': 6},
    {'subject': 'Linear Algebra and Calculus', 'description': 'Gauss Elimination, Row Echelon Form, and Rank of a Matrix', 'min_study_time': 3, 'max_study_time': 5, 'priority': 4, 'prerequisite': 'Systems of Linear Equations', 'deadline': 5},
    {'subject': 'Linear Algebra and Calculus', 'description': 'Fundamental Theorem for Linear Systems', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': 'Gauss Elimination, Row Echelon Form, and Rank of a Matrix', 'deadline': 7},
    {'subject': 'Linear Algebra and Calculus', 'description': 'Eigenvalues and Eigenvectors', 'min_study_time': 3, 'max_study_time': 5, 'priority': 4, 'prerequisite': 'Fundamental Theorem for Linear Systems', 'deadline': 5},
    {'subject': 'Linear Algebra and Calculus', 'description': 'Diagonalization of Matrices', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': 'Eigenvalues and Eigenvectors', 'deadline': 7},
    {'subject': 'Linear Algebra and Calculus', 'description': 'Orthogonal Transformation, Quadratic Forms, and Canonical Forms', 'min_study_time': 3, 'max_study_time': 5, 'priority': 4, 'prerequisite': 'Diagonalization of Matrices', 'deadline': 8},

    # Module 2: Multivariable Calculus - Differentiation
    {'subject': 'Linear Algebra and Calculus', 'description': 'Limit and Continuity of Functions of Two Variables', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': None, 'deadline': 6},
    {'subject': 'Linear Algebra and Calculus', 'description': 'Partial Derivatives, Chain Rule, and Total Derivative', 'min_study_time': 3, 'max_study_time': 5, 'priority': 4, 'prerequisite': 'Limit and Continuity of Functions of Two Variables', 'deadline': 5},
    {'subject': 'Linear Algebra and Calculus', 'description': 'Relative Maxima and Minima', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': 'Partial Derivatives, Chain Rule, and Total Derivative', 'deadline': 7},
    {'subject': 'Linear Algebra and Calculus', 'description': 'Absolute Maxima and Minima on Closed and Bounded Set', 'min_study_time': 3, 'max_study_time': 5, 'priority': 4, 'prerequisite': 'Relative Maxima and Minima', 'deadline': 6},

    # Module 3: Multivariable Calculus - Integration
    {'subject': 'Linear Algebra and Calculus', 'description': 'Double Integrals (Cartesian)', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': None, 'deadline': 6},
    {'subject': 'Linear Algebra and Calculus', 'description': 'Reversing the Order of Integration', 'min_study_time': 3, 'max_study_time': 5, 'priority': 4, 'prerequisite': 'Double Integrals (Cartesian)', 'deadline': 7},
    {'subject': 'Linear Algebra and Calculus', 'description': 'Change of Coordinates (Cartesian to Polar)', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': 'Reversing the Order of Integration', 'deadline': 8},
    {'subject': 'Linear Algebra and Calculus', 'description': 'Finding Areas and Volume Using Double Integrals', 'min_study_time': 3, 'max_study_time': 5, 'priority': 4, 'prerequisite': 'Change of Coordinates (Cartesian to Polar)', 'deadline': 6},
    {'subject': 'Linear Algebra and Calculus', 'description': 'Mass and Centre of Gravity of Inhomogeneous Laminas Using Double Integral', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': 'Finding Areas and Volume Using Double Integrals', 'deadline': 7},
    {'subject': 'Linear Algebra and Calculus', 'description': 'Triple Integrals', 'min_study_time': 3, 'max_study_time': 5, 'priority': 4, 'prerequisite': 'Mass and Centre of Gravity of Inhomogeneous Laminas Using Double Integral', 'deadline': 8},
    {'subject': 'Linear Algebra and Calculus', 'description': 'Volume Calculated as Triple Integral', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': 'Triple Integrals', 'deadline': 6},
    {'subject': 'Linear Algebra and Calculus', 'description': 'Triple Integral in Cylindrical and Spherical Coordinates', 'min_study_time': 3, 'max_study_time': 5, 'priority': 4, 'prerequisite': 'Volume Calculated as Triple Integral', 'deadline': 7},
    {'subject': 'Linear Algebra and Calculus', 'description': 'Computations Involving Spheres, Cylinders', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': 'Triple Integral in Cylindrical and Spherical Coordinates', 'deadline': 8},

    # Module 4: Sequences and Series
    {'subject': 'Linear Algebra and Calculus', 'description': 'Convergence of Sequences and Series', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': None, 'deadline': 6},
    {'subject': 'Linear Algebra and Calculus', 'description': 'Convergence of Geometric Series and P-Series', 'min_study_time': 3, 'max_study_time': 5, 'priority': 4, 'prerequisite': 'Convergence of Sequences and Series', 'deadline': 5},
    {'subject': 'Linear Algebra and Calculus', 'description': 'Test of Convergence (Comparison, Ratio, and Root Tests)', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': 'Convergence of Geometric Series and P-Series', 'deadline': 7},
    {'subject': 'Linear Algebra and Calculus', 'description': 'Alternating Series and Leibnitz Test', 'min_study_time': 3, 'max_study_time': 5, 'priority': 4, 'prerequisite': 'Test of Convergence (Comparison, Ratio, and Root Tests)', 'deadline': 6},
    {'subject': 'Linear Algebra and Calculus', 'description': 'Absolute and Conditional Convergence', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': 'Alternating Series and Leibnitz Test', 'deadline': 8},

    # Module 5: Series Representation of Functions
    {'subject': 'Linear Algebra and Calculus', 'description': 'Taylor Series', 'min_study_time': 3, 'max_study_time': 5, 'priority': 4, 'prerequisite': None, 'deadline': 5},
    {'subject': 'Linear Algebra and Calculus', 'description': 'Binomial Series and Series Representation of Exponential, Trigonometric, Logarithmic Functions', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': 'Taylor Series', 'deadline': 7},
    {'subject': 'Linear Algebra and Calculus', 'description': 'Fourier Series', 'min_study_time': 3, 'max_study_time': 5, 'priority': 4, 'prerequisite': 'Binomial Series and Series Representation of Exponential, Trigonometric, Logarithmic Functions', 'deadline': 8},
    {'subject': 'Linear Algebra and Calculus', 'description': 'Euler Formulas', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': 'Fourier Series', 'deadline': 6},
    {'subject': 'Linear Algebra and Calculus', 'description': 'Convergence of Fourier Series', 'min_study_time': 3, 'max_study_time': 5, 'priority': 4, 'prerequisite': 'Euler Formulas', 'deadline': 7},
    {'subject': 'Linear Algebra and Calculus', 'description': 'Half Range Sine and Cosine Series, Parseval’s Theorem', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': 'Convergence of Fourier Series', 'deadline': 8},
    
    
      # Module 1: Electrochemistry and Corrosion
    {'subject': 'Engineering Chemistry', 'description': 'Introduction to Electrochemistry and Differences between Electrolytic and Electrochemical Cells', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': None, 'deadline': 6},
    {'subject': 'Engineering Chemistry', 'description': 'Electrochemical Cells and Redox Reactions', 'min_study_time': 3, 'max_study_time': 5, 'priority': 4, 'prerequisite': 'Introduction to Electrochemistry and Differences between Electrolytic and Electrochemical Cells', 'deadline': 5},
    {'subject': 'Engineering Chemistry', 'description': 'Cell Representation and Types of Electrodes', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': 'Electrochemical Cells and Redox Reactions', 'deadline': 6},
    {'subject': 'Engineering Chemistry', 'description': 'Reference Electrodes and SHE', 'min_study_time': 3, 'max_study_time': 5, 'priority': 4, 'prerequisite': 'Cell Representation and Types of Electrodes', 'deadline': 5},
    # ... Continue adding topics for Module 1

    # Module 2: Spectroscopic Techniques and Applications
    {'subject': 'Engineering Chemistry', 'description': 'Introduction to Spectroscopic Techniques and Types of Spectrum', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': None, 'deadline': 6},
    {'subject': 'Engineering Chemistry', 'description': 'UV-Visible Spectroscopy Principles and Applications', 'min_study_time': 3, 'max_study_time': 5, 'priority': 4, 'prerequisite': 'Introduction to Spectroscopic Techniques and Types of Spectrum', 'deadline': 5},
    {'subject': 'Engineering Chemistry', 'description': 'Molecular Energy Levels and Beer Lambert’s Law', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': 'UV-Visible Spectroscopy Principles and Applications', 'deadline': 6},
    {'subject': 'Engineering Chemistry', 'description': 'IR-Spectroscopy Principles and Applications', 'min_study_time': 3, 'max_study_time': 5, 'priority': 4, 'prerequisite': 'Molecular Energy Levels and Beer Lambert’s Law', 'deadline': 5},
    {'subject': 'Engineering Chemistry', 'description': '1H NMR Spectroscopy Principles and Applications', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': 'IR-Spectroscopy Principles and Applications', 'deadline': 6},
    # ... Continue adding topics for Module 2

    # Module 3: Instrumental Methods and Nanomaterials
    {'subject': 'Engineering Chemistry', 'description': 'Principles of Thermal Analysis and Applications', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': None, 'deadline': 6},
    {'subject': 'Engineering Chemistry', 'description': 'Chromatographic Methods and Nanomaterials', 'min_study_time': 3, 'max_study_time': 5, 'priority': 4, 'prerequisite': 'Principles of Thermal Analysis and Applications', 'deadline': 5},
    {'subject': 'Engineering Chemistry', 'description': 'TGA and DTA Principles and Applications', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': 'Chromatographic Methods and Nanomaterials', 'deadline': 6},
    {'subject': 'Engineering Chemistry', 'description': 'Basic Principles and Applications of Column Chromatography', 'min_study_time': 3, 'max_study_time': 5, 'priority': 4, 'prerequisite': 'TGA and DTA Principles and Applications', 'deadline': 5},
    {'subject': 'Engineering Chemistry', 'description': 'GC and HPLC Principles and Applications', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': 'Basic Principles and Applications of Column Chromatography', 'deadline': 6},
    # ... Continue adding topics for Module 3

    # Module 4: Stereochemistry and Polymer Chemistry
    {'subject': 'Engineering Chemistry', 'description': 'Isomerism and Representation of 3D Structures', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': None, 'deadline': 6},
    {'subject': 'Engineering Chemistry', 'description': 'Stereoisomerism and Conformational Analysis', 'min_study_time': 3, 'max_study_time': 5, 'priority': 4, 'prerequisite': 'Isomerism and Representation of 3D Structures', 'deadline': 5},
    {'subject': 'Engineering Chemistry', 'description': 'Geometrical Isomerism and Optical Isomerism', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': 'Stereoisomerism and Conformational Analysis', 'deadline': 6},
    {'subject': 'Engineering Chemistry', 'description': 'R-S Notation and Optical Isomerism', 'min_study_time': 3, 'max_study_time': 5, 'priority': 4, 'prerequisite': 'Geometrical Isomerism and Optical Isomerism', 'deadline': 5},
    # ... Continue adding topics for Module 4

    # Module 5: Water Chemistry and Sewage Water Treatment
    {'subject': 'Engineering Chemistry', 'description': 'Water Characteristics and Hardness', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': None, 'deadline': 6},
    {'subject': 'Engineering Chemistry', 'description': 'Water Softening Methods and Reverse Osmosis', 'min_study_time': 3, 'max_study_time': 5, 'priority': 4, 'prerequisite': 'Water Characteristics and Hardness', 'deadline': 5},
    {'subject': 'Engineering Chemistry', 'description': 'Disinfection Methods and Municipal Water Treatment', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': 'Water Softening Methods and Reverse Osmosis', 'deadline': 6},
    {'subject': 'Engineering Chemistry', 'description': 'Dissolved Oxygen and Water Treatment', 'min_study_time': 3, 'max_study_time': 5, 'priority': 4, 'prerequisite': 'Disinfection Methods and Municipal Water Treatment', 'deadline': 5},
    {'subject': 'Engineering Chemistry', 'description': 'Sewage Water Treatment and Flow Diagrams', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': 'Dissolved Oxygen and Water Treatment', 'deadline': 6},
    # ... Continue adding topics for Module 5
        
      {'subject': 'Basics of Civil & Mechanical Engineering', 'description': 'Relevance of Civil Engineering in Infrastructural Development', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': None, 'deadline': 6},
    {'subject': 'Basics of Civil & Mechanical Engineering', 'description': 'Responsibility of an Engineer in Ensuring Safety', 'min_study_time': 3, 'max_study_time': 5, 'priority': 4, 'prerequisite': 'Relevance of Civil Engineering in Infrastructural Development', 'deadline': 5},
    {'subject': 'Basics of Civil & Mechanical Engineering', 'description': 'Introduction to Major Disciplines of Civil Engineering', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': 'Responsibility of an Engineer in Ensuring Safety', 'deadline': 6},
    # ... Continue adding topics for Module 1

    # Module 2: Surveying and Construction Materials
    {'subject': 'Basics of Civil & Mechanical Engineering', 'description': 'Importance, Objectives, and Principles of Surveying', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': None, 'deadline': 6},
    {'subject': 'Basics of Civil & Mechanical Engineering', 'description': 'Conventional Construction Materials: Bricks, Stones, Cement, Sand, and Timber', 'min_study_time': 3, 'max_study_time': 5, 'priority': 4, 'prerequisite': 'Importance, Objectives, and Principles of Surveying', 'deadline': 5},
    {'subject': 'Basics of Civil & Mechanical Engineering', 'description': 'Cement Concrete and Steel: Constituents, Properties, and Types', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': 'Conventional Construction Materials: Bricks, Stones, Cement, Sand, and Timber', 'deadline': 6},
    # ... Continue adding topics for Module 2

    # Module 3: Building Construction and Basic Infrastructure Services
    {'subject': 'Basics of Civil & Mechanical Engineering', 'description': 'Foundations: Bearing Capacity, Functions, and Types', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': None, 'deadline': 6},
    {'subject': 'Basics of Civil & Mechanical Engineering', 'description': 'Brick Masonry and Roofs/Floors: Types and Functions', 'min_study_time': 3, 'max_study_time': 5, 'priority': 4, 'prerequisite': 'Foundations: Bearing Capacity, Functions, and Types', 'deadline': 5},
    {'subject': 'Basics of Civil & Mechanical Engineering', 'description': 'Basic Infrastructure Services: MEP, HVAC, Elevators, etc.', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': 'Brick Masonry and Roofs/Floors: Types and Functions', 'deadline': 6},
    # ... Continue adding topics for Module 3

    # Module 4: Analysis of Thermodynamic Cycles and IC Engines
    {'subject': 'Basics of Civil & Mechanical Engineering', 'description': 'Analysis of Thermodynamic Cycles: Carnot, Otto, Diesel', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': None, 'deadline': 6},
    {'subject': 'Basics of Civil & Mechanical Engineering', 'description': 'IC Engines: CI, SI, 2-Stroke, 4-Stroke', 'min_study_time': 3, 'max_study_time': 5, 'priority': 4, 'prerequisite': 'Analysis of Thermodynamic Cycles: Carnot, Otto, Diesel', 'deadline': 5},
    {'subject': 'Basics of Civil & Mechanical Engineering', 'description': 'Efficiencies of IC Engines and Systems', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': 'IC Engines: CI, SI, 2-Stroke, 4-Stroke', 'deadline': 6},   
    ]

# Convert list of dictionaries to CSV file
list_of_dict_to_csv(data, 'data.csv')

# Convert CSV file to list of dictionaries
new_data = csv_to_list_of_dict('data.csv')
print(new_data)
