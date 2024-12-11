import numpy as np


def jd_to_angle(jd):
    """
    Takes in the time as a Julian Date and
    returns the angle between the First Point of Aries and the Greenwich Meridian

    Input:
    param jd: time in units of Julian Date

    Output:
    param theta: angle between the prime meridian and the first point of aries, in units of radians
    """
    # We use equations 1.44-1.47 provided on pp. 23 of the Chapter 1 lecture notes 
    # to compute the angle theta from the Julian date:
    t = jd - 2451545.0  # 1.44 (days elapsed since January 1.0, 2000)
    T = t/36525.  # 1.45
    theta = np.mod(280.46061837 + (360.98564736629 * t) + (0.0003879332 * (T ** 2)) - ((T ** 3)/38710000.), 
                   360.)  # 1.46 and 1.47, computes theta in units of degrees (mod 360)
    # return theta in units of radians
    return np.deg2rad(theta)


def true_anomaly_to_ecc_anomaly(e, f):
    """
    Converts the eccentricity e and the true anomaly f, in radians, to the eccentric anomaly E
    also in radians

    Input:
    param e: eccentricity
    param f: true anomaly, units of radians

    Output:
    param E: eccentric anomaly, units of radians
    """
    # if e is not in (0, 1] then raise error
    if (e < 0) or (e > 1):
        raise ValueError("Eccentricity must be in the range (0, 1]")
    # compute eccentric anomaly
    return 2 * np.arctan(np.sqrt((1 - e)/(1 + e)) * np.tan(f/2))  
    # return np.arctan((np.sin(f) * np.sqrt(1 - (e ** 2)))/(np.cos(f) + e))


def ecc_anomaly_to_true_anomaly(e, E):
    """
    Converts the eccentricity e and the eccentric anomaly E, in radians, to the true anomaly f
    also in radians

    Input:
    param e: eccentricity
    param E: eccentric anomaly, units of radians

    Output:
    param f: true anomaly, units of radians
    """
    # if e is not in (0, 1] then raise error
    if (e <= 0) or (e > 1):
        raise ValueError("Eccentricity must be in the range (0, 1]")
    # compute true anomaly
    return 2 * np.arctan(np.sqrt((1 + e)/(1 - e)) * np.tan(E/2))   


def aef_to_xyz(a, e, f, mu):
    """
    Converts the semi-major axis a, eccentricity e, and true anomaly f 
    of a body, m2, orbiting a central mass, m1, in the two-body problem
    to its cartesian x, y, z position and velocity in the orbital plane.

    Input:
    param a: semi-major axis, arbitrary units of distance
    param e: eccentricity
    param f: true anomaly, radians
    param mu: standard gravitational parameter G * (m1 + m2), 
    arbitrary units of distance^3 and time^-2 but units of distance must match those of a

    Output:
    Tuple of:
    param pos: list of [x, y, z] Cartesian components of position, same units of distance as a
    param vel: list of [xdot, ydot, zdot] Cartesian components of velocity, same units of distance and time as a and mu
    """
    # raise errors for invalid inputs:
    # semi-major axis
    if not a > 0:
        raise ValueError("Semi-major axis must be greater than zero")
    # eccentricity will be raised by ecc_anomaly_to_true_anomaly function
    # mu
    if not mu > 0:
        raise ValueError("Standard gravitational parameter must be greater than zero")
    # Convert true anomaly to eccentric anomaly using the subroutine defined in Question 3a):
    E = true_anomaly_to_ecc_anomaly(e, f)
    # Compute x, y, and z using equations (2), (3), (4) in the assignment
    x = a * (np.cos(E) - e)
    y = a * np.sqrt(1 - (e ** 2)) * np.sin(E)
    z = 0.
    # Compute the mean motion n which has units of rad/time
    n = np.sqrt(mu/(a ** 3))
    # Compute xdot, ydot and zdot using equations (5), (6), (7) in the assignment
    xdot = -(a * n * np.sin(E))/(1 - e * np.cos(E))
    ydot = (a * np.sqrt(1 - (e ** 2)) * n * np.cos(E))/(1 - e * np.cos(E))
    zdot = 0.
    # return position and velocity
    pos = [x, y, z]
    vel = [xdot, ydot, zdot]
    return pos, vel


def rotate_pos_vel_vectors(pos, vel, euler_angles):
    """
    Rotates the Cartesian position and velocity of an object relative to a central mass in the orbital plane
    to the heliocentric ecliptic frame by its orbital Euler angles
    
    Input:
    param pos: list of x, y, z: Cartesian position of object, arbitrary units
    param vel: list of xdot, ydot, zdot: Cartesian velocity of object, arbitrary units
    param euler_angles: list/tuple of the Euler angles:
                        omega (arg. of pericenter), 
                        i (inclination), 
                        Omega (long. of ascending node), 
                        all in radians, in that order

    Output:
    Tuple of:
    param pos_rotated: list of [x, y, z]: rotated position vector in the heliocentric ecliptic frame, same units as pos
    param vel_rotated: list of [xdot, ydot, zdot]: rotated velocity vector in the heliocentric ecliptic frame, same units as vel
    """
    # Unpack list/tuple of Euler angles
    omega, inc, Omega = euler_angles
    # Use the rotation matrix specified above to rotate the position and velocity vectors above
    R = np.array([
        [np.cos(Omega) * np.cos(omega) - np.sin(Omega) * np.cos(inc) * np.sin(omega), -np.cos(Omega) * np.sin(omega) - np.sin(Omega) * np.cos(inc) * np.cos(omega), np.sin(Omega) * np.sin(inc)],
        [np.sin(Omega) * np.cos(omega) + np.cos(Omega) * np.cos(inc) * np.sin(omega), -np.sin(Omega) * np.sin(omega) + np.cos(Omega) * np.cos(inc) * np.cos(omega), -np.cos(Omega) * np.sin(inc)],
        [np.sin(inc) * np.sin(omega), np.sin(inc) * np.cos(omega), np.cos(inc)]
    ])
    # perform rotations by multiplying by the rotation matrix
    pos_rotated = R @ np.array(pos)
    vel_rotated = R @ np.array(vel)
    # return rotated positions and velocities
    return list(pos_rotated), list(vel_rotated)


def orbital_elements_to_cart_helio(mu, a, e, i, Omega, omega, f):
    """
    Computes the heliocentric ecliptic Cartesian positions and velocities of a particle of mass m2 orbiting a central body of mass m1
    given the gravitational parameter mu and the standard orbital elements
    
    Input:
    param mu: gravitational parameter G * (m1 + m2), arbitrary units of distance^3 and time^-2
    param a: semi-major axis, arbitrary units of distance but must match those of mu
    param e: eccentricity
    param i: inclination, radians
    param Omega: longitude of the ascending node, radians 
    param omega: argument of pericenter, radians
    param f: true anomaly, radians

    Output:
    Tuple of:
    param pos_helio: list of [x, y, z]: position vector of m2 in the heliocentric ecliptic frame, same units of distance as in mu
    param vel_helio: list of [xdot, ydot, zdot]: velociy vector of m2 in the heliocentric ecliptic frame, same units of distance and time as for mu
    """
    # first compute the cartesian position and velocity in the orbital plane
    pos_plane, vel_plane = aef_to_xyz(a, e, f, mu)
    # then rotate from the orbital plane to heliocentric ecliptic frame
    pos_helio, vel_helio = rotate_pos_vel_vectors(pos_plane, vel_plane, [omega, i, Omega])
    # return heliocentric position and velocity
    return pos_helio, vel_helio