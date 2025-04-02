def calculate_coefficients(n_terms):
    """
    Calculate coefficients for the power series expansion of f(x) = 2/(3x^2-4x+1)
    using the recurrence relation.
    """
    a = [0] * n_terms
    a[0] = 2  # First coefficient
    
    if n_terms > 1:
        a[1] = 4 * a[0]  # Second coefficient: 4*2 = 8
    
    # Apply recurrence relation: a_n = 4*a_{n-1} - 3*a_{n-2}
    for i in range(2, n_terms):
        a[i] = 4 * a[i-1] - 3 * a[i-2]
    
    return a

# Calculate and print the first 10 coefficients
coefficients = calculate_coefficients(11)
for n, coef in enumerate(coefficients):
    print(f"Coefficient of x^{n}: {coef}")

# Verify the coefficient of x^5
print(f"\nCoefficient of x^5: {coefficients[5]}")