function y = applyOperatorShifted(x, shift, h, wind)
% The convection-diffusion stencil of demo_operator_forms, applied in place and
% taking its parameters as trailing arguments -- which is what subgmres
% forwards. A file of its own rather than a local function of the script:
% MATLAB wants those at the end, Octave only sees them once execution has
% passed the definition, and no placement satisfies both.
y = (2/h^2 + shift)*x ...
	+ [(-1/h^2 + wind/(2*h))*x(2:end); 0] ...
	+ [0; (-1/h^2 - wind/(2*h))*x(1:end-1)];
end
