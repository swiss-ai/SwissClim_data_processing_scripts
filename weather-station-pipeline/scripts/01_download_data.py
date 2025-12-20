import cdsapi
from datetime import datetime, timedelta
import os
import time

# Initialize the CDS client
client = cdsapi.Client()

# Create output directory if it doesn't exist
output_dir = 'station_data'
os.makedirs(output_dir, exist_ok=True)

# Configuration
wait_seconds = 5  # Wait between requests
start_date = datetime(2000, 1, 1)
end_date = datetime(2024, 12, 31)
current_date = start_date

# Counter for statistics
downloaded_count = 0
skipped_count = 0
error_count = 0

print(f"Starting download for year 2023...")
print(f"Output directory: {output_dir}")
print(f"Wait time between requests: {wait_seconds} seconds\n")

while current_date <= end_date:
    year = current_date.strftime('%Y')
    month = current_date.strftime('%m')
    day = current_date.strftime('%d')

    # Create output filename: station_year_month_day.nc
    output_filename = f'raw_data/station_{year}_{month}_{day}.nc'
    output_path = os.path.join(output_dir, output_filename)

    # Skip if already downloaded
    if os.path.exists(output_path):
        print(f"⊘ Already exists: {output_filename}")
        skipped_count += 1
        current_date += timedelta(days=1)
        continue

    print(f"Downloading data for {year}-{month}-{day}...")

    try:
        request = client.retrieve(
            'insitu-observations-surface-land',
            {
                'version': '2_0_0',
                'time_aggregation': 'sub_daily',
                'variable': [
                    'accumulated_precipitation',
                    'air_pressure',
                    'air_pressure_at_sea_level',
                    'air_temperature',
                    'dew_point_temperature',
                    'fresh_snow',
                    'snow_depth',
                    'snow_water_equivalent',
                    'wind_from_direction',
                    'wind_speed'
                ],
                'year': year,
                'month': month,
                'day': day,
                'data_format': 'netcdf'
            },
            output_path
        )
        print(f"✓ Downloaded: {output_filename}")
        downloaded_count += 1
        time.sleep(wait_seconds)  # Wait before next request
    except Exception as e:
        print(f"✗ Error downloading {year}-{month}-{day}: {e}")
        error_count += 1

    # Move to next day
    current_date += timedelta(days=1)

# Print summary
print("\n" + "="*60)
print("Download Summary:")
print(f"  Downloaded: {downloaded_count} files")
print(f"  Skipped (already exist): {skipped_count} files")
print(f"  Errors: {error_count} files")
print(f"  Total days processed: {downloaded_count + skipped_count + error_count}")
print("="*60)
print("Download complete!")
