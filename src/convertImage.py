'''
__author__: Abhishek Thakur
'''


ref_X =  95.047
ref_Y = 100.000
ref_Z = 108.883

def xyz_to_cielab(X, Y, Z):
    '''
    Convert from XYZ to CIE-L*a*b*
    '''
    var_X = X / ref_X
    var_Y = Y / ref_Y
    var_Z = Z / ref_Z

    if var_X > 0.008856:
        var_X **= ( 1./3. )
    else:
        var_X = ( 7.787 * var_X ) + ( 16. / 116. )
    if var_Y > 0.008856:
        var_Y **= ( 1./3. )
    else:
        var_Y = ( 7.787 * var_Y ) + ( 16. / 116. )
    if var_Z > 0.008856:
        var_Z **= ( 1./3. )
    else:
        var_Z = ( 7.787 * var_Z ) + ( 16. / 116. )

    CIE_L = ( 116 * var_Y ) - 16.
    CIE_a = 500. * ( var_X - var_Y )
    CIE_b = 200. * ( var_Y - var_Z )

    return CIE_L, CIE_a, CIE_b

def rgb_to_xyz(R, G, B):
    '''
    Convert from RGB to XYZ.
    '''
    var_R = ( R / 255.)
    var_G = ( G / 255.)
    var_B = ( B / 255.)

    if var_R > 0.04045:
        var_R = ( ( var_R + 0.055 ) / 1.055 ) ** 2.4
    else:
        var_R /= 12.92

    if var_G > 0.04045:
        var_G = ( ( var_G + 0.055 ) / 1.055 ) ** 2.4
    else:
        var_G /= 12.92
    if var_B > 0.04045:
        var_B = ( ( var_B + 0.055 ) / 1.055 ) ** 2.4
    else:
        var_B /= 12.92

    var_R *= 100
    var_G *= 100
    var_B *= 100

    #Observer. = 2 deg, Illuminant = D65
    X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805
    Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722
    Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505

    return X,Y,Z


def rgb2lab(R,G,B):
    '''
    Convert from RGB to CIE-L*a*b*.
    '''
    X,Y,Z = rgb_to_xyz(R,G,B)
    return xyz_to_cielab(X,Y,Z)