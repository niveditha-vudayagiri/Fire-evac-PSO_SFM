import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import csv
import os

class StatisticsCollector:
    def __init__(self, pedestrians):
        self.pedestrians = pedestrians
        self.stats = {}

    def generate_report(self, algorithm_name="SFM", csv_path=None, dashboard_data=None):
        evac_times = [p.evac_time for p in self.pedestrians if getattr(p, 'evacuated', False)]
        total_evacuated = len(evac_times)
        avg_time = sum(evac_times) / total_evacuated if total_evacuated else 0
        total = len(self.pedestrians)
        deaths = [p for p in self.pedestrians if getattr(p, 'dead', False) or (not getattr(p, 'evacuated', False) and getattr(p, 'evac_time', None) is None)]
        total_deaths = len(deaths)
        # Deaths per floor
        deaths_per_floor = Counter(getattr(p, 'floor', -1) for p in deaths)
        # Staff vs no staff
        staff_deaths = sum(1 for p in deaths if getattr(p, 'is_staff', False))
        nonstaff_deaths = total_deaths - staff_deaths
        staff_survived = sum(1 for p in self.pedestrians if getattr(p, 'is_staff', False) and getattr(p, 'evacuated', False))
        nonstaff_survived = sum(1 for p in self.pedestrians if not getattr(p, 'is_staff', False) and getattr(p, 'evacuated', False))
        # Age group survival
        age_groups = defaultdict(lambda: {'survived': 0, 'dead': 0})
        for p in self.pedestrians:
            age = getattr(p, 'age', None)
            if age is not None:
                if getattr(p, 'evacuated', False):
                    age_groups[self._age_group(age)]['survived'] += 1
                else:
                    age_groups[self._age_group(age)]['dead'] += 1
        # Store stats for dashboard
        self.stats = {
            'algorithm': algorithm_name,
            'total': total,
            'evacuated': total_evacuated,
            'deaths': total_deaths,
            'avg_time': avg_time,
            'deaths_per_floor': dict(deaths_per_floor),
            'staff_survived': staff_survived,
            'staff_deaths': staff_deaths,
            'nonstaff_survived': nonstaff_survived,
            'nonstaff_deaths': nonstaff_deaths,
            'age_groups': dict(age_groups),
        }
        # Print report
        print(f"\n[{algorithm_name} Report]")
        print(f"Total Pedestrians: {total}")
        print(f"Total Evacuated: {total_evacuated}")
        print(f"Total Deaths: {total_deaths}")
        print(f"Average Evacuation Time: {avg_time:.2f} seconds")
        print("Deaths per Floor:")
        for floor, count in sorted(deaths_per_floor.items()):
            print(f"  Floor {floor}: {count}")
        print(f"Staff Survived: {staff_survived}, Staff Deaths: {staff_deaths}")
        print(f"Non-Staff Survived: {nonstaff_survived}, Non-Staff Deaths: {nonstaff_deaths}")
        print("Age Group Survival:")
        for group, stats in age_groups.items():
            print(f"  {group}: Survived={stats['survived']}, Dead={stats['dead']}")
        # CSV export
        if csv_path:
            self.export_csv(csv_path, algorithm_name)
        # Dashboard integration
        if dashboard_data is not None:
            dashboard_data[algorithm_name] = self.stats
        #self.plot_evacuation_curve(evac_times, algorithm_name)
        self.plot_distribution(evac_times, algorithm_name)

    def export_csv(self, csv_path, algorithm_name):
        # Write summary stats and per-pedestrian data
        fieldnames = [
            'algorithm', 'ped_id', 'evacuated', 'evac_time', 'dead', 'floor', 'is_staff', 'age', 'age_group'
        ]
        write_header = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for p in self.pedestrians:
                writer.writerow({
                    'algorithm': algorithm_name,
                    'ped_id': getattr(p, 'id', None),
                    'evacuated': getattr(p, 'evacuated', False),
                    'evac_time': getattr(p, 'evac_time', None),
                    'dead': getattr(p, 'dead', False),
                    'floor': getattr(p, 'floor', None),
                    'is_staff': getattr(p, 'is_staff', False),
                    'age': getattr(p, 'age', None),
                    'age_group': self._age_group(getattr(p, 'age', 0)) if getattr(p, 'age', None) is not None else None
                })

    def _age_group(self, age):
        if age < 13:
            return 'Child'
        elif age < 60:
            return 'Adult'
        else:
            return 'Elderly'

    def plot_evacuation_curve(self, evac_times, label="Algorithm"):
        evac_times.sort()
        x = list(range(1, len(evac_times)+1))
        y = evac_times
        plt.figure(figsize=(8, 4))
        plt.plot(y, x, label=label)
        plt.xlabel('Time (s)')
        plt.ylabel('Number of People Evacuated')
        plt.title('Evacuation Curve')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_distribution(self, evac_times, label="Algorithm"):
        plt.figure(figsize=(8, 4))
        plt.hist(evac_times, bins=20, color='skyblue', edgecolor='black')
        plt.xlabel('Evacuation Time (s)')
        plt.ylabel('Number of People')
        plt.title(f'Evacuation Time Distribution ({label})')
        plt.grid(True)
        plt.show()
