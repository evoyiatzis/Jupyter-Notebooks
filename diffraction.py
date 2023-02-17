#!/usr/bin/env python3

"""Defines a dictionary with the scattering functions for all elements
   Key is a string with the element
   Value is the scattering function as described in Brown P J, Fox A G, Maslen E N,
   O'Keefe M A and Willis B T M 2004 Intensity of diffraction intensities
   International Tables for Crystallography Volume C: Mathematical, Physical, and Chemical Tables
   ed E Prince (Norwell, MA: Kluwer Academic Publishers) pp 55495
"""

import os
import sys
from argparse import ArgumentParser
import numpy as np
from diffraction_library import scattering_factor

def create_cli():
    '''Create an elementary CLI based on the argparse module & perform initial consistency checks'''

    # set a minimal  command line interface
    parser = ArgumentParser(description='Computing XRD patterns for given structures')

    parser.add_argument("input_file", help="The full path & name of the input XYZ file.\
                        The coordinates should be in Angstrom")
    parser.add_argument("output_file", help="The full path & name of the output file.\
                        It contains the scattering angle in radians and the diffraction pattern")
    parser.add_argument("-min","--minimum_angle", default=0, type=float, help="Minimum \
                         scattering angle in radians")
    parser.add_argument("-max","--maximum_angle", default=np.pi, type=float, help="Maximum \
                         scattering angle in radians")
    parser.add_argument("-n","--number_of_points", default=10, type=int, help="How many points\
                        to use between the minimum and maximum angle")
    parser.add_argument("-w","--wavelength", default=1.5418, type=float, help="The wavelength \
                        of the incident electrons in Angstrom")

    arguments = parser.parse_args()

    # check that the specified input file exists
    if not os.path.exists(arguments.input_file):
        print("The specified input file does not exist. The script will exit")
        sys.exit()

    # check if the specified output file exists and issue a warning
    if os.path.exists(arguments.output_file):
        print("The specified output file already exists. The script will overwrite it")

    if arguments.number_of_points < 1:
        print("The number of points must be a positive integer")
        sys.exit()

    # check that the wavelength is a positive number
    if arguments.wavelength <= 0:
        print("The wavelength must be a positive number. The script will exit")
        sys.exit()

    return arguments

def compute_diffraction(sigmas, elmnts, xcoords, ycoords, zcoords):
    '''The function performing the actual calculation '''

    scattering = np.zeros(sigmas.size)

    # find the unique elements in the structure
    unique_elements = set(elmnts)

    # get the scattering from the dictionary for each element type
    scattering_dict = {}
    for ielement in unique_elements:
        try:
            scattering_dict[ielement] = scattering_factor[ielement](sigmas)
        except KeyError:
            print("The element {ielement} has no scattering function. The script will exit")
            sys.exit()

    # compute the distances between all pairs of particles
    dist_x = np.transpose(np.tile(xcoords, (len(xcoords),1))) - xcoords
    dist_y = np.transpose(np.tile(ycoords, (len(xcoords),1))) - ycoords
    dist_z = np.transpose(np.tile(zcoords, (len(xcoords),1))) - zcoords

    distance = np.sqrt(np.square(dist_x) + np.square(dist_y) + np.square(dist_z))

    # compute the prefactors for the scattering
    prefactor_vector = np.zeros(xcoords.size)

    for iscatter, s_value in enumerate(sigmas):

        # form the factor with the atomic scattering values
        for ielement in unique_elements:
            prefactor_vector[elmnts == ielement] = scattering_dict[ielement][iscatter]

        prefactor_matrix = np.outer(prefactor_vector, prefactor_vector)

        # compute the sinusoidal term
        sin_term = np.true_divide(np.sin(4*np.pi*s_value*distance), 4*np.pi*s_value*distance, \
                where=(s_value*distance != 0) )
        np.fill_diagonal(sin_term, 0)

        # multiply the factor with the intermediate matrix
        final = np.multiply(prefactor_matrix, sin_term)

        # sum the upper half of the matrix
        scattering[iscatter] = 0.5*final.sum()

    return scattering

def read_xyz_file(input_file):
    '''Read a particle's configuration from an input file '''

    with open(input_file, "r", encoding="utf-8") as ifile:
        natoms = int(ifile.readline().split()[0])
        ifile.readline()
        in_data = np.genfromtxt(ifile, dtype=None, encoding="utf-8", names=['element','x','y','z'])
        if natoms != in_data.size:
            print("The specified number of atoms differs from the number of coordinates")
            sys.exit()

    return (in_data['element'], in_data['x'], in_data['y'], in_data['z'])

def powder_diffraction():
    '''The main function for computing the diffraction pattern of a given structure '''

    # get input-output files & parameters from user
    args = create_cli()

    # first step read the structure from an input XYZ file
    elements, x_coords, y_coords, z_coords = read_xyz_file(args.input_file)

    # second step compute the actual scattering pattern
    thetas = np.linspace(args.minimum_angle, args.maximum_angle, args.number_of_points)
    sigmas = np.sin(thetas)/args.wavelength
    scatter = compute_diffraction(sigmas, elements, x_coords, y_coords, z_coords)

    # third step write the computed data to the output file
    np.savetxt(args.output_file, np.c_[thetas, scatter])

if __name__ == "__main__":
    powder_diffraction()
