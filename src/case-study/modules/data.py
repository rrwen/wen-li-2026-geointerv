import geopandas as gpd
import numpy as np

from shapely.geometry import box, Polygon

# Generate rectangular grids using a bounding box
def gen_grid(bounds, cells=None, cell_width=None, cell_height=None):
    
    # Get bounds or extract if gdf
    xmin, ymin, xmax, ymax = bounds.total_bounds if isinstance(bounds, gpd.GeoDataFrame) else bounds

    # Get crs if exists
    crs = bounds.crs if isinstance(bounds, gpd.GeoDataFrame) else None
    
    # Create grid based on num of cells
    # Edited from James Brennan's code
    # https://james-brennan.github.io/posts/fast_gridding_geopandas/
    if cells:

        # Calculate cell size based on num of cells
        cell_size = (xmax-xmin)/cells
        
        # Generate cells
        grid_cells = []
        for x0 in np.arange(xmin, xmax+cell_size, cell_size):
            for y0 in np.arange(ymin, ymax+cell_size, cell_size):
                x1 = x0-cell_size
                y1 = y0+cell_size
                grid_cells.append(box(x0, y0, x1, y1))
        
        # Format into gdf based on num of cells
        out = gpd.GeoDataFrame(
            grid_cells,
            columns=['geometry'],
            crs=crs
        )
        out.index = list(out.index)

    # Create grid based on cell size
    # Edited from user Mativane's code
    # https://gis.stackexchange.com/questions/269243/creating-polygon-grid-using-geopandas
    else:
    
        # Calculate rows and cols
        cols = list(np.arange(xmin, xmax + cell_width, cell_width))
        rows = list(np.arange(ymin, ymax + cell_height, cell_height))

        # Generate grid based on bounds, cols, and rows
        polygons = []
        for x in cols[:-1]:
            for y in rows[:-1]:
                polygons.append(Polygon([(x,y), (x + cell_width, y), (x + cell_width, y + cell_height), (x, y + cell_height)]))

        # Format into gdf based on cell size
        out = gpd.GeoDataFrame({'geometry': polygons}, crs=crs)
        out.index = list(out.index)

    return out
