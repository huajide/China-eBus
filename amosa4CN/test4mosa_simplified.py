import multiprocessing as mp
from functools import partial
import os
import pandas as pd
from vehicle_type import battery_capacity
import data_utils
from dateutil import parser


def process_single_city(city_name, cities_file='../data/18cities.csv',
                        input_dir='../data/input/vs_parking_nodeid',
                        output_dir='../data/input/vs_parking_nodeid_simplified'):
    """Simplify parking-trip data for a single city."""
    try:
        print(f"Processing city: {city_name}")

        # Create the output directory.
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Skip the city if the simplified file already exists.
        output_file = os.path.join(output_dir, f"{city_name}.csv")
        if os.path.exists(output_file):
            print(f"  Simplified file already exists, skipping: {city_name}")
            return True

        # Read the input parking-trip data.
        input_file = os.path.join(input_dir, f"{city_name}.csv")
        if not os.path.exists(input_file):
            print(f"  Input file not found, skipping: {input_file}")
            return False

        vs_parking_df = pd.read_csv(input_file)
        print(f"  Loaded {len(vs_parking_df)} parking records")

        # Parse time fields.
        vs_parking_df['s_time'] = vs_parking_df['s_time'].apply(parser.parse)
        vs_parking_df['e_time'] = vs_parking_df['e_time'].apply(parser.parse)

        # Use the minimum battery capacities as simplification parameters.
        min_range_large = min(battery_capacity[0])
        min_range_medium = min(battery_capacity[1])
        min_range_small = min(battery_capacity[2])

        print(f"  Using minimum battery capacities as simplification parameters: {min_range_large}/{min_range_medium}/{min_range_small} kwh")

        # Simplify the vehicle-trip table with data_utils.simplify_vs_df.
        print(f"  Starting simplification...")
        vs_parking_df_simplified = data_utils.simplify_vs_df(vs_parking_df,
                                                             min_range_large, min_range_medium, min_range_small)
        print(f"  {len(vs_parking_df_simplified)} records remain after simplification")

        # Save the simplified dataset.
        vs_parking_df_simplified.to_csv(output_file, index=False)
        print(f"  Simplified data saved to: {output_file}")

        return True

    except Exception as e:
        print(f"  Error processing city {city_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def generate_simplified_parking_data_multiprocess(cities_file='../data/224cities.csv',
                                                  input_dir='../data/input/vs_parking_nodeid',
                                                  output_dir='../data/input/vs_parking_nodeid_simplified',
                                                  num_processes=None):
    """Generate simplified parking-trip data for all cities with multiprocessing."""
    # Read the city list.
    cities = pd.read_csv(cities_file)
    city_names = cities['city'].tolist()
    print(f"Total cities to process: {len(city_names)}")

    # Create the output directory.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Fix shared parameters with functools.partial.
    process_func = partial(process_single_city,
                           cities_file=cities_file,
                           input_dir=input_dir,
                           output_dir=output_dir)

    # Process cities in parallel.
    if num_processes is None:
        num_processes = mp.cpu_count()

    print(f"Processing with {num_processes} workers")

    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_func, city_names)

    # Summarize the results.
    success_count = sum(results)
    print(f"\nCompleted. Successfully processed {success_count}/{len(city_names)} cities")
    print(f"Simplified data saved in: {output_dir}")


# Example usage
if __name__ == '__main__':
    # Batch-process all cities with multiprocessing.
    print("=== Batch generation of simplified parking-trip data ===")
    generate_simplified_parking_data_multiprocess()
