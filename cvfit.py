from increasingF2 import increasingF2
import numpy as np

#_match_cumulative_cdf() was taken from skimage.exposure code, but modified to return interpolator function/transform function
#instead of the transformed source.
def _match_cumulative_cdf(source, template):
    """
    Return modified source array so that the cumulative density function of
    its values matches the cumulative density function of the template.
    """
    src_values, src_unique_indices, src_counts = np.unique(source.ravel(), return_inverse=True, return_counts=True)
    tmpl_values, tmpl_counts = np.unique(template.ravel(), return_counts=True)

    # calculate normalized quantiles for each array
    src_quantiles = np.cumsum(src_counts) / source.size
    tmpl_quantiles = np.cumsum(tmpl_counts) / template.size

    f = interpolate(src_quantiles, tmpl_quantiles)
    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
    return interp_a_values[src_unique_indices].reshape(source.shape)

def cvfit(x,y,method='quad'):
    mask = (x > 0) & (y > 0)
    if method == 'quad':
        mapf = increasingF2(x[mask],y[mask], p = 1e-5)
        if( mapf == None ):
            sys.stderr.write("WARN: cvfit(): Failed to find optimized mapping function using quadratic programming approach.  Trying histogram approach.")
            method = 'hist'
    if method == 'hist':
        mapf = _match_cumulative_cdf(x[msk],y[msk])
    return(mapf)
