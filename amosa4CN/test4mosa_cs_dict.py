import multiprocessing as mp
import pandas as pd
import geopandas as gpd
import pickle
import os
from pyproj import CRS
import networkx as nx
import data_utils

def generate_all_d2s_dict_for_city(city_name, sim_n=50):
    """Generate all_d2s_dict for a single city."""
    try:
        print(f"Processing city: {city_name}")

        # Read vehicle-trip data.
        vs_parking_df = pd.read_csv(f'../data/input/vs_parking_nodeid/{city_name}.csv')

        # Read charging-station data.
        cs_gdf = gpd.read_file(f'../data/input/cs_gdf/{city_name}.shp', crs=CRS.from_epsg(4547))

        # Read road-network data.
        nodes_sim = gpd.read_file(f'../data/road/{city_name}/nodes_sim.shp', crs=CRS.from_epsg(4547))
        edges_sim = gpd.read_file(f'../data/road/{city_name}/edges_sim.shp', crs=CRS.from_epsg(4547))

        # Build the graph.
        G = nx.from_pandas_edgelist(df=edges_sim, source='u', target='v',
                                    edge_attr=['edge_id', 'length'], create_using=nx.Graph())

        # Generate all_d2s_dict.
        all_d2s_dict = data_utils.get_d2s_realdict(
            vs_parking_df, cs_gdf, nodes_sim, G,
            near_n=150, sim_n=sim_n, distance_limit=100000.0, is_projected=True
        )

        # Save the result.
        output_dir = f"../data/input/all_d2s_dict"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(f"{output_dir}/{city_name}.pkl", 'wb') as f:
            pickle.dump(all_d2s_dict, f)

        print(f"Finished processing city: {city_name}")
        return f"City {city_name} processed successfully"

    except Exception as e:
        print(f"Error processing city {city_name}: {str(e)}")
        return f"City {city_name} failed: {str(e)}"

def batch_generate_all_d2s_dict(city_list, sim_n=50, num_processes=None):
    """Generate all_d2s_dict for multiple cities with multiprocessing."""
    if num_processes is None:
        num_processes = mp.cpu_count()

    print(f"Processing {len(city_list)} cities with {num_processes} workers")

    # Create the process pool.
    with mp.Pool(processes=num_processes) as pool:
        # Process all cities in the pool.
        results = pool.starmap(generate_all_d2s_dict_for_city,
                               [(city, sim_n) for city in city_list])

    # Summarize the processing results.
    success_count = sum(1 for result in results if "processed successfully" in result)
    fail_count = len(results) - success_count

    print(f"\nProcessing summary:")
    print(f"Success: {success_count} cities")
    print(f"Failed: {fail_count} cities")

    return results

def get_cities_without_all_d2s_dict(all_cities_list):
    """Return the list of cities without generated all_d2s_dict files."""
    all_d2s_dir = "../data/input/all_d2s_dict"

    if not os.path.exists(all_d2s_dir):
        return all_cities_list

    existing_files = os.listdir(all_d2s_dir)
    existing_cities = [f.replace('.pkl', '') for f in existing_files if f.endswith('.pkl')]

    missing_cities = [city for city in all_cities_list if city not in existing_cities]
    return missing_cities

# Example usage
if __name__ == '__main__':
    # Read the full city list.
    cities_df = pd.read_csv(rf'../data/224cities.csv')  # rf'../data/224cities.csv'
    all_cities = cities_df['city'].tolist()

    # Find cities that still need processing.
    cities_to_process = get_cities_without_all_d2s_dict(all_cities)

    if cities_to_process:
        print(f"Cities to process: {cities_to_process}")
        # Run batch generation.
        results = batch_generate_all_d2s_dict(cities_to_process, sim_n=80, num_processes=16)
    else:
        print("all_d2s_dict already exists for all cities")
