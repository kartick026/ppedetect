#!/usr/bin/env python3
"""
Automated PPE Compliance Reporting System
Generate comprehensive safety reports
"""

import json
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
from pathlib import Path
import numpy as np

class PPEComplianceReporter:
    def __init__(self, data_dir="compliance_data"):
        """Initialize compliance reporting system"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.data_dir / "reports").mkdir(exist_ok=True)
        (self.data_dir / "data").mkdir(exist_ok=True)
        (self.data_dir / "charts").mkdir(exist_ok=True)
        
        print(f"[INFO] Compliance reporting system initialized")
        print(f"[INFO] Data directory: {self.data_dir}")
    
    def generate_daily_report(self, date=None):
        """Generate daily compliance report"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        print(f"\n[INFO] Generating daily report for {date}")
        
        # Load detection data
        detections = self.load_detection_data(date)
        
        if not detections:
            print(f"[WARNING] No detection data found for {date}")
            return None
        
        # Calculate metrics
        metrics = self.calculate_metrics(detections)
        
        # Generate report
        report = {
            'date': date,
            'total_detections': len(detections),
            'compliance_rate': metrics['compliance_rate'],
            'helmet_detections': metrics['helmet_count'],
            'safety_vest_detections': metrics['safety_vest_count'],
            'goggles_detections': metrics['goggles_count'],
            'gloves_detections': metrics['gloves_count'],
            'non_compliant_instances': metrics['non_compliant_count'],
            'peak_violation_times': metrics['peak_violation_times'],
            'recommendations': self.generate_recommendations(metrics)
        }
        
        # Save report
        report_file = self.data_dir / "reports" / f"daily_report_{date}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate charts
        self.generate_daily_charts(detections, date)
        
        print(f"[SUCCESS] Daily report generated: {report_file}")
        return report
    
    def generate_weekly_report(self, week_start=None):
        """Generate weekly compliance report"""
        if week_start is None:
            week_start = datetime.now() - timedelta(days=7)
        
        print(f"\n[INFO] Generating weekly report for week starting {week_start.strftime('%Y-%m-%d')}")
        
        # Collect daily data for the week
        weekly_data = []
        for i in range(7):
            date = (week_start + timedelta(days=i)).strftime("%Y-%m-%d")
            detections = self.load_detection_data(date)
            if detections:
                weekly_data.extend(detections)
        
        if not weekly_data:
            print("[WARNING] No data found for the week")
            return None
        
        # Calculate weekly metrics
        metrics = self.calculate_metrics(weekly_data)
        
        # Generate weekly trends
        trends = self.calculate_weekly_trends(week_start)
        
        # Create report
        report = {
            'week_start': week_start.strftime("%Y-%m-%d"),
            'week_end': (week_start + timedelta(days=6)).strftime("%Y-%m-%d"),
            'total_detections': len(weekly_data),
            'average_daily_detections': len(weekly_data) / 7,
            'compliance_rate': metrics['compliance_rate'],
            'helmet_detections': metrics['helmet_count'],
            'safety_vest_detections': metrics['safety_vest_count'],
            'goggles_detections': metrics['goggles_count'],
            'gloves_detections': metrics['gloves_count'],
            'non_compliant_instances': metrics['non_compliant_count'],
            'trends': trends,
            'recommendations': self.generate_weekly_recommendations(metrics, trends)
        }
        
        # Save report
        report_file = self.data_dir / "reports" / f"weekly_report_{week_start.strftime('%Y-%m-%d')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate weekly charts
        self.generate_weekly_charts(weekly_data, week_start)
        
        print(f"[SUCCESS] Weekly report generated: {report_file}")
        return report
    
    def generate_monthly_report(self, month=None, year=None):
        """Generate monthly compliance report"""
        if month is None:
            month = datetime.now().month
        if year is None:
            year = datetime.now().year
        
        print(f"\n[INFO] Generating monthly report for {year}-{month:02d}")
        
        # Collect all data for the month
        monthly_data = []
        for day in range(1, 32):  # Check all possible days
            try:
                date = datetime(year, month, day).strftime("%Y-%m-%d")
                detections = self.load_detection_data(date)
                if detections:
                    monthly_data.extend(detections)
            except ValueError:
                break  # Invalid date (e.g., Feb 30)
        
        if not monthly_data:
            print("[WARNING] No data found for the month")
            return None
        
        # Calculate monthly metrics
        metrics = self.calculate_metrics(monthly_data)
        
        # Generate monthly trends
        trends = self.calculate_monthly_trends(year, month)
        
        # Create comprehensive report
        report = {
            'month': f"{year}-{month:02d}",
            'total_detections': len(monthly_data),
            'average_daily_detections': len(monthly_data) / 30,
            'compliance_rate': metrics['compliance_rate'],
            'helmet_detections': metrics['helmet_count'],
            'safety_vest_detections': metrics['safety_vest_count'],
            'goggles_detections': metrics['goggles_count'],
            'gloves_detections': metrics['gloves_count'],
            'non_compliant_instances': metrics['non_compliant_count'],
            'trends': trends,
            'top_violation_patterns': self.identify_violation_patterns(monthly_data),
            'recommendations': self.generate_monthly_recommendations(metrics, trends)
        }
        
        # Save report
        report_file = self.data_dir / "reports" / f"monthly_report_{year}-{month:02d}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate monthly charts
        self.generate_monthly_charts(monthly_data, year, month)
        
        print(f"[SUCCESS] Monthly report generated: {report_file}")
        return report
    
    def load_detection_data(self, date):
        """Load detection data for a specific date"""
        data_file = self.data_dir / "data" / f"detections_{date}.json"
        
        if not data_file.exists():
            return []
        
        try:
            with open(data_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load data for {date}: {e}")
            return []
    
    def calculate_metrics(self, detections):
        """Calculate compliance metrics from detection data"""
        if not detections:
            return {
                'compliance_rate': 0,
                'helmet_count': 0,
                'safety_vest_count': 0,
                'goggles_count': 0,
                'gloves_count': 0,
                'non_compliant_count': 0,
                'peak_violation_times': []
            }
        
        total_detections = len(detections)
        compliant_detections = 0
        ppe_counts = {'helmet': 0, 'safety_vest': 0, 'goggles': 0, 'gloves': 0}
        violation_times = []
        
        for detection in detections:
            detected_classes = set()
            if 'detections' in detection:
                for det in detection['detections']:
                    ppe_type = det['class']
                    ppe_counts[ppe_type] += 1
                    detected_classes.add(ppe_type)
            
            # Check compliance (requires helmet and safety_vest)
            if 'helmet' in detected_classes and 'safety_vest' in detected_classes:
                compliant_detections += 1
            else:
                violation_times.append(detection.get('timestamp', ''))
        
        compliance_rate = (compliant_detections / total_detections * 100) if total_detections > 0 else 0
        
        return {
            'compliance_rate': compliance_rate,
            'helmet_count': ppe_counts['helmet'],
            'safety_vest_count': ppe_counts['safety_vest'],
            'goggles_count': ppe_counts['goggles'],
            'gloves_count': ppe_counts['gloves'],
            'non_compliant_count': total_detections - compliant_detections,
            'peak_violation_times': violation_times[:10]  # Top 10 violation times
        }
    
    def calculate_weekly_trends(self, week_start):
        """Calculate weekly trends"""
        trends = {}
        for i in range(7):
            date = (week_start + timedelta(days=i)).strftime("%Y-%m-%d")
            detections = self.load_detection_data(date)
            if detections:
                metrics = self.calculate_metrics(detections)
                trends[date] = {
                    'compliance_rate': metrics['compliance_rate'],
                    'total_detections': len(detections)
                }
        return trends
    
    def calculate_monthly_trends(self, year, month):
        """Calculate monthly trends"""
        trends = {}
        for day in range(1, 32):
            try:
                date = datetime(year, month, day).strftime("%Y-%m-%d")
                detections = self.load_detection_data(date)
                if detections:
                    metrics = self.calculate_metrics(detections)
                    trends[date] = {
                        'compliance_rate': metrics['compliance_rate'],
                        'total_detections': len(detections)
                    }
            except ValueError:
                break
        return trends
    
    def identify_violation_patterns(self, detections):
        """Identify common violation patterns"""
        patterns = {
            'missing_helmet_only': 0,
            'missing_vest_only': 0,
            'missing_both': 0,
            'time_patterns': {}
        }
        
        for detection in detections:
            if 'detections' in detection:
                detected_classes = {det['class'] for det in detection['detections']}
                has_helmet = 'helmet' in detected_classes
                has_vest = 'safety_vest' in detected_classes
                
                if not has_helmet and not has_vest:
                    patterns['missing_both'] += 1
                elif not has_helmet:
                    patterns['missing_helmet_only'] += 1
                elif not has_vest:
                    patterns['missing_vest_only'] += 1
                
                # Time pattern analysis
                timestamp = detection.get('timestamp', '')
                if timestamp:
                    hour = datetime.fromisoformat(timestamp).hour
                    if hour not in patterns['time_patterns']:
                        patterns['time_patterns'][hour] = 0
                    patterns['time_patterns'][hour] += 1
        
        return patterns
    
    def generate_recommendations(self, metrics):
        """Generate recommendations based on metrics"""
        recommendations = []
        
        if metrics['compliance_rate'] < 80:
            recommendations.append("Compliance rate is below 80%. Consider additional safety training.")
        
        if metrics['safety_vest_count'] < metrics['helmet_count']:
            recommendations.append("Safety vest compliance is lower than helmet compliance. Focus on vest awareness.")
        
        if metrics['non_compliant_count'] > 0:
            recommendations.append("Non-compliant instances detected. Review safety protocols.")
        
        return recommendations
    
    def generate_weekly_recommendations(self, metrics, trends):
        """Generate weekly recommendations"""
        recommendations = self.generate_recommendations(metrics)
        
        # Analyze trends
        compliance_rates = [day['compliance_rate'] for day in trends.values()]
        if len(compliance_rates) > 1:
            if compliance_rates[-1] < compliance_rates[0]:
                recommendations.append("Compliance rate declining over the week. Immediate attention needed.")
        
        return recommendations
    
    def generate_monthly_recommendations(self, metrics, trends):
        """Generate monthly recommendations"""
        recommendations = self.generate_recommendations(metrics)
        
        # Monthly-specific recommendations
        if metrics['compliance_rate'] > 90:
            recommendations.append("Excellent compliance rate! Consider sharing best practices.")
        
        return recommendations
    
    def generate_daily_charts(self, detections, date):
        """Generate daily compliance charts"""
        if not detections:
            return
        
        # Create compliance timeline
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Daily PPE Compliance Report - {date}', fontsize=16)
        
        # Compliance rate over time
        times = []
        compliance_rates = []
        
        for detection in detections:
            if 'timestamp' in detection:
                times.append(datection['timestamp'])
                # Calculate compliance for this detection
                detected_classes = set()
                if 'detections' in detection:
                    detected_classes = {det['class'] for det in detection['detections']}
                
                is_compliant = 'helmet' in detected_classes and 'safety_vest' in detected_classes
                compliance_rates.append(100 if is_compliant else 0)
        
        if times and compliance_rates:
            axes[0, 0].plot(range(len(times)), compliance_rates, 'b-', linewidth=2)
            axes[0, 0].set_title('Compliance Rate Over Time')
            axes[0, 0].set_xlabel('Detection Number')
            axes[0, 0].set_ylabel('Compliance Rate (%)')
            axes[0, 0].grid(True, alpha=0.3)
        
        # PPE detection counts
        ppe_counts = {'helmet': 0, 'safety_vest': 0, 'goggles': 0, 'gloves': 0}
        for detection in detections:
            if 'detections' in detection:
                for det in detection['detections']:
                    ppe_type = det['class']
                    if ppe_type in ppe_counts:
                        ppe_counts[ppe_type] += 1
        
        axes[0, 1].bar(ppe_counts.keys(), ppe_counts.values(), color=['orange', 'yellow', 'blue', 'green'])
        axes[0, 1].set_title('PPE Detection Counts')
        axes[0, 1].set_ylabel('Count')
        
        # Compliance pie chart
        compliant = sum(1 for d in detections if self.is_compliant(d))
        non_compliant = len(detections) - compliant
        
        axes[1, 0].pie([compliant, non_compliant], labels=['Compliant', 'Non-Compliant'], 
                      colors=['green', 'red'], autopct='%1.1f%%')
        axes[1, 0].set_title('Overall Compliance')
        
        # Hourly distribution
        hourly_counts = {}
        for detection in detections:
            if 'timestamp' in detection:
                hour = datetime.fromisoformat(detection['timestamp']).hour
                hourly_counts[hour] = hourly_counts.get(hour, 0) + 1
        
        if hourly_counts:
            hours = sorted(hourly_counts.keys())
            counts = [hourly_counts[h] for h in hours]
            axes[1, 1].bar(hours, counts, color='skyblue')
            axes[1, 1].set_title('Detection Distribution by Hour')
            axes[1, 1].set_xlabel('Hour of Day')
            axes[1, 1].set_ylabel('Detection Count')
        
        plt.tight_layout()
        chart_file = self.data_dir / "charts" / f"daily_charts_{date}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] Daily charts saved: {chart_file}")
    
    def generate_weekly_charts(self, detections, week_start):
        """Generate weekly compliance charts"""
        # Similar to daily charts but aggregated by day
        pass
    
    def generate_monthly_charts(self, detections, year, month):
        """Generate monthly compliance charts"""
        # Similar to daily charts but aggregated by day
        pass
    
    def is_compliant(self, detection):
        """Check if a detection is compliant"""
        if 'detections' not in detection:
            return False
        
        detected_classes = {det['class'] for det in detection['detections']}
        return 'helmet' in detected_classes and 'safety_vest' in detected_classes
    
    def export_to_csv(self, report_type, date_range):
        """Export compliance data to CSV"""
        # Implementation for CSV export
        pass

def main():
    """Main function for compliance reporting"""
    print("="*70)
    print("PPE COMPLIANCE REPORTING SYSTEM")
    print("="*70)
    
    reporter = PPEComplianceReporter()
    
    print("\n[INFO] Available report types:")
    print("1. Daily report")
    print("2. Weekly report") 
    print("3. Monthly report")
    print("4. All reports")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice in ["1", "4"]:
        # Generate daily report
        today = datetime.now().strftime("%Y-%m-%d")
        reporter.generate_daily_report(today)
    
    if choice in ["2", "4"]:
        # Generate weekly report
        week_start = datetime.now() - timedelta(days=7)
        reporter.generate_weekly_report(week_start)
    
    if choice in ["3", "4"]:
        # Generate monthly report
        reporter.generate_monthly_report()
    
    print("\n[SUCCESS] Compliance reporting complete!")
    print(f"[INFO] Reports saved in: {reporter.data_dir / 'reports'}")
    print(f"[INFO] Charts saved in: {reporter.data_dir / 'charts'}")

if __name__ == "__main__":
    main()
