These are the key procedures involved in the implementation process of article "An Automated Framework for Analyzing Structural Evolution in On-the-fly Non-adiabatic Molecular Dynamics Using Autoencoder and Multiple Molecular Descriptors", including obtaining three descriptors from molecular XYZ coordinate (from_xyz_to_mbtr_netinput.py, from_xyz_to_soap_netinput.py, from_xyz_to_aev_netinput.py), dimensionality reduction by Autoencoder (Autoencoder.py) and calculating the information entropy for each internal coordinate (Information_Entropy.py)

For "from_xyz_to_mbtr_netinput.py", "from_xyz_to_soap_netinput.py" and "from_xyz_to_aev_netinput.py", they are codes for generating the MBTR, SOAP and AEV descriptors respectively. The xyz file needs to be prepared for each molecule to be analyzed. And there is an additional configuration file, "Atomic_Environment_Vectors.py", prepared for the AEV descriptor.

For "Autoencoder.py", it is a code that uses Autoencoder for dimensionality reduction. Its input is the matrix of initial data.

For "Information_Entropy.py", it used to calculate the information entropy for each internal coordinate. Please prepare the redundant internal corrdinate matrices for both the initial and hopping geometries of each cluster of molecule respectively.
