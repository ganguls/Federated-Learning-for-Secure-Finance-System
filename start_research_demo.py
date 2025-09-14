#!/usr/bin/env python3
"""
Quick Start Script for Research Demonstration
Final Year Project - Federated Learning Attack Detection
"""

import subprocess
import time
import webbrowser
import sys
import os
from pathlib import Path

def print_banner():
    """Print the demonstration banner"""
    print("=" * 70)
    print("🎓 FEDERATED LEARNING ATTACK DETECTION - RESEARCH DEMO")
    print("=" * 70)
    print("Final Year Project Presentation System")
    print("Data Poisoning Prevention in Federated Learning")
    print("=" * 70)
    print()

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'flask', 'flask_socketio', 'flask_cors', 'numpy', 
        'pandas', 'scikit-learn', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All dependencies are available!")
    return True

def run_comprehensive_demo():
    """Run the comprehensive demonstration script"""
    print("\n🚀 Running comprehensive demonstration...")
    
    try:
        # Run the demo presentation script
        result = subprocess.run([
            sys.executable, 'demo_presentation.py'
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Comprehensive demo completed successfully!")
            print("📊 Results saved to 'demo_results' folder")
            return True
        else:
            print(f"❌ Demo failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Demo timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        return False

def start_dashboard():
    """Start the enhanced dashboard"""
    print("\n🌐 Starting enhanced dashboard...")
    
    try:
        # Change to dashboard directory
        dashboard_dir = Path("dashboard")
        if not dashboard_dir.exists():
            print("❌ Dashboard directory not found!")
            return None
        
        # Start the dashboard
        process = subprocess.Popen([
            sys.executable, "app.py"
        ], cwd=dashboard_dir)
        
        # Wait a moment for the server to start
        time.sleep(3)
        
        print("✅ Dashboard started successfully!")
        print("🌐 Main Dashboard: http://localhost:5000")
        print("🎓 Research Demo: http://localhost:5000/research")
        
        return process
        
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        return None

def open_browser():
    """Open browser to the research demo page"""
    print("\n🌐 Opening research demonstration page...")
    
    try:
        webbrowser.open("http://localhost:5000/research")
        print("✅ Browser opened to research demo page")
    except Exception as e:
        print(f"⚠️  Could not open browser automatically: {e}")
        print("Please manually open: http://localhost:5000/research")

def print_instructions():
    """Print demonstration instructions"""
    print("\n" + "=" * 70)
    print("📋 DEMONSTRATION INSTRUCTIONS")
    print("=" * 70)
    print()
    print("1. 🎯 Research Demo Page:")
    print("   - URL: http://localhost:5000/research")
    print("   - Interactive attack detection demonstration")
    print("   - Real-time visualization and metrics")
    print()
    print("2. 🔧 Available Features:")
    print("   - Multiple attack types (Label Flipping, Gradient Poisoning, etc.)")
    print("   - Adjustable privacy levels (ε values)")
    print("   - Configurable malicious client percentages")
    print("   - Comprehensive demo across multiple scenarios")
    print("   - Results export for documentation")
    print()
    print("3. 📊 Key Metrics to Highlight:")
    print("   - Detection Accuracy: Overall correctness")
    print("   - F1 Score: Balance of precision and recall")
    print("   - Privacy-Detection Trade-off: Impact of ε values")
    print("   - Attack Type Performance: Different scenarios")
    print()
    print("4. 🎓 Presentation Tips:")
    print("   - Start with basic detection demo")
    print("   - Show privacy level impact")
    print("   - Demonstrate different attack types")
    print("   - Run comprehensive analysis")
    print("   - Export and discuss results")
    print()
    print("5. 📁 Generated Files:")
    print("   - demo_results/: Comprehensive analysis results")
    print("   - research_results_*.json: Exported data")
    print("   - presentation_guide.md: Detailed presentation guide")
    print()
    print("=" * 70)
    print("🎉 Ready for your final year project presentation!")
    print("=" * 70)

def main():
    """Main function"""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies and try again.")
        return
    
    # Run comprehensive demo
    print("\n" + "=" * 50)
    print("STEP 1: Running Comprehensive Analysis")
    print("=" * 50)
    
    if not run_comprehensive_demo():
        print("\n⚠️  Comprehensive demo failed, but dashboard will still work.")
    
    # Start dashboard
    print("\n" + "=" * 50)
    print("STEP 2: Starting Enhanced Dashboard")
    print("=" * 50)
    
    dashboard_process = start_dashboard()
    if not dashboard_process:
        print("\n❌ Failed to start dashboard. Please check the error messages above.")
        return
    
    # Open browser
    print("\n" + "=" * 50)
    print("STEP 3: Opening Demonstration Interface")
    print("=" * 50)
    
    open_browser()
    
    # Print instructions
    print_instructions()
    
    # Keep the script running
    try:
        print("\n🔄 Dashboard is running. Press Ctrl+C to stop.")
        print("📖 Check presentation_guide.md for detailed instructions.")
        dashboard_process.wait()
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping dashboard...")
        dashboard_process.terminate()
        print("✅ Dashboard stopped. Good luck with your presentation!")

if __name__ == "__main__":
    main()

