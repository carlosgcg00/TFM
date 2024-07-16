def fractional_bits_required(number, precision=10):
    """
    Calculate the number of bits required to represent the fractional part of a number 
    in binary with a specified precision.
    
    Args:
    number (float): The number whose fractional part is to be represented in binary.
    precision (int): The maximum number of bits to consider for the fractional part (default is 10).
    
    Returns:
    int: The number of bits required to represent the fractional part.
    """
    
    # Get the fractional part of the number
    fractional_part = number - int(number)
    
    # If the fractional part is zero, no bits are required
    if fractional_part == 0:
        return 0
    
    # List to store the binary representation of the fractional part
    binary_fractional_part = []
    
    # Repeat until the fractional part is 0 or we reach the desired precision
    while fractional_part != 0 and len(binary_fractional_part) < precision:
        fractional_part *= 2
        bit = int(fractional_part)
        binary_fractional_part.append(bit)
        fractional_part -= bit
    
    # The number of bits required is the length of the binary_fractional_part list
    return len(binary_fractional_part)



def convert_fractional_to_binary(number, precision=32):
    fractional_part = number - int(number)
    binary_fractional_part = []
    while fractional_part != 0 and len(binary_fractional_part) < precision:
        fractional_part *= 2
        bit = int(fractional_part)
        binary_fractional_part.append(abs(bit))
        fractional_part -= bit
    return binary_fractional_part

def fractional_bits_required(number, precision=10):
  
    # The number of bits required is the length of the binary_fractional_part list
    return len(convert_fractional_to_binary(number, precision=precision))

def convert_fractional_binary_to_fractional(int_value, binary_fractional_part):
    fractional_part = abs(int_value)
    for i in range(len(binary_fractional_part)):
        fractional_part += abs(binary_fractional_part[i]) * 2**(-(i+1))
    return fractional_part

def convert_int_binary_to_int(binary_int_part):
    fractional_part = 0
    for i in range(len(binary_int_part)):
        fractional_part += binary_int_part[i] * 2**(i)
    return fractional_part




def string_bit(sign_bit, int_bits, frac_bits):
    
    bits = f'{1 if sign_bit==-1 else 0}'
    for int_bit in int_bits:
        bits = f'{bits}{int_bit}'
    bits = f'{bits}.'
    for frac_bit in frac_bits:
        bits = f'{bits}{frac_bit}'
    return bits