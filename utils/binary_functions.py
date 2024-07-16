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