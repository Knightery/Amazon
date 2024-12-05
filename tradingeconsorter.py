import datetime
import re

def parse_path_data():
    # Step 1: Gather inputs
    path_data = input("Enter the path data: ")
    start_date = input("Enter the start date (YYYY-MM-DD): ")
    end_date = input("Enter the end date (YYYY-MM-DD): ")
    start_rate = float(input("Enter the starting interest rate (%): "))
    end_rate = float(input("Enter the ending interest rate (%): "))
    
    # Step 2: Extract `L x y` pairs from the path data
    pattern = r"L\s([\d.]+)\s([\d.]+)"
    matches = re.findall(pattern, path_data)
    points = [(float(x), float(y)) for x, y in matches]

    # Include the initial `M x y` point (if present)
    initial_point_match = re.match(r"M\s([\d.]+)\s([\d.]+)", path_data)
    if initial_point_match:
        initial_point = (float(initial_point_match.group(1)), float(initial_point_match.group(2)))
        points.insert(0, initial_point)

    # Determine start and end numbers (y-values)
    start_number = points[0][1]
    end_number = points[-1][1]
    
    # Determine total `x` range for proportional time calculation
    total_x = points[-1][0]

    # Step 3: Map proportions of `x` to time range
    start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end_date - start_date).days
    date_intervals = []
    current_date = start_date

    for i, (x, y) in enumerate(points):
        if i == 0:
            continue  # Skip the initial point since it's the start date
        proportion = x / total_x
        days = int(round(proportion * total_days))
        next_date = start_date + datetime.timedelta(days=days)
        date_intervals.append((current_date.date(), next_date.date()))
        current_date = next_date

    # Step 4: Map chart numbers (`y`) to interest rates
    rate_difference = end_rate - start_rate
    chart_difference = start_number - end_number  # Chart values decrease as rates increase
    rate_per_unit = rate_difference / chart_difference

    rates = [
        start_rate - rate_per_unit * (value - start_number)
        for _, value in points
    ]

    # Step 5: Combine dates and rates, filter for the last data point of each month
    result = [
    {"date": next_date, "interest_rate": rate}
    for (_, next_date), rate in zip(date_intervals, rates[1:])  # Skip the first rate since it's the initial point
    ]

    monthly_result = []
    last_date_by_month = {}

    for entry in result:
        date = entry["date"]
        month_key = (date.year, date.month)
        last_date_by_month[month_key] = entry

    # Ensure all months from start to end are represented
    current_month = start_date.replace(day=1)
    while current_month <= end_date:
        month_key = (current_month.year, current_month.month)
        if month_key in last_date_by_month:
            monthly_result.append(last_date_by_month[month_key])
        else:
            # Fill missing months with the same rate as the previous month
            if monthly_result:
                previous_rate = monthly_result[-1]["interest_rate"]
                monthly_result.append({"date": current_month.date(), "interest_rate": previous_rate})
        current_month += datetime.timedelta(days=32)
        current_month = current_month.replace(day=1)

    # Step 6: Output result
    for entry in monthly_result:
        print(f"Date: {entry['date']}, Rate: {entry['interest_rate']:.2f}%")

# Run the script
if __name__ == "__main__":
    parse_path_data()
