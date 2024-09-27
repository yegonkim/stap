import numpy as np

def laplacian_periodic(u, dx):
    """
    Approximates the Laplacian using periodic boundary conditions.
    
    Args:
        u: The function values at each grid point (array of length nx).
        dx: The grid spacing (assuming uniform grid).
        
    Returns:
        uxx: The second derivative (Laplacian) of u with periodic boundary conditions.
    """
    uxx = np.zeros_like(u)
    
    # Compute second derivative using central difference, with periodic boundary conditions
    uxx[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
    
    # Periodic boundary conditions
    uxx[0] = (u[1] - 2 * u[0] + u[-1]) / dx**2  # Left boundary wraps to the rightmost point
    uxx[-1] = (u[0] - 2 * u[-1] + u[-2]) / dx**2  # Right boundary wraps to the leftmost point
    
    return uxx

def allen_cahn_deriv(u, dx, xi, nu):
    """
    Computes the time derivative of the Allen-Cahn equation with periodic boundary conditions.
    
    Args:
        u: The current state values (array of length nx).
        dx: The grid spacing (assuming uniform grid).
        xi: Interface thickness parameter.
        nu: Diffusion coefficient.
        
    Returns:
        dudt: The time derivative of u.
    """
    # Compute second derivative (Laplacian) with periodic boundary conditions
    uxx = laplacian_periodic(u, dx)
    
    # Allen-Cahn time derivative
    dudt = nu * uxx - (u * (u**2 - 1)) / (2 * xi**2)
    
    return dudt

def allen_cahn_next_step(u, dx, xi, nu, dt):
    """
    Computes the next time step for the Allen-Cahn equation using Euler's method.
    
    Args:
        u: The current state values (array of length nx).
        dx: The grid spacing.
        xi: Interface thickness parameter.
        nu: Diffusion coefficient.
        dt: The time step size.
        
    Returns:
        u_next: The state values at the next time step.
    """
    # Compute the time derivative (du/dt)
    dudt = allen_cahn_deriv(u, dx, xi, nu)
    
    # Compute the next state values using Euler's method
    u_next = u + dudt * dt
    
    return u_next
