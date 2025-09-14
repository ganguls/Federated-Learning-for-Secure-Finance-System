#!/usr/bin/env python3
"""
Docker-specific startup script for Research Demonstration
Final Year Project - Federated Learning Attack Detection
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def print_banner():
    """Print the demonstration banner"""
    print("=" * 70)
    print("🎓 FEDERATED LEARNING ATTACK DETECTION - DOCKER DEMO")
    print("=" * 70)
    print("Final Year Project Presentation System")
    print("Data Poisoning Prevention in Federated Learning")
    print("Running in Docker Container")
    print("=" * 70)
    print()

def check_environment():
    """Check Docker environment"""
    print("🔍 Checking Docker environment...")
    
    # Check if we're in a Docker container
    if os.path.exists('/.dockerenv'):
        print("  ✅ Running in Docker container")
    else:
        print("  ⚠️  Not running in Docker container")
    
    # Check Python environment
    print(f"  ✅ Python version: {sys.version}")
    
    # Check working directory
    print(f"  ✅ Working directory: {os.getcwd()}")
    
    # Check if demo files exist
    demo_files = [
        'demo_presentation.py',
        'start_research_demo.py',
        'presentation_guide.md',
        'RESEARCH_PROJECT_SUMMARY.md'
    ]
    
    for file in demo_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file}")
    
    print("✅ Environment check completed!")
    return True

def run_comprehensive_demo():
    """Run the comprehensive demonstration"""
    print("\n🚀 Running comprehensive demonstration...")
    
    try:
        # Run the demo presentation script
        result = subprocess.run([
            sys.executable, 'demo_presentation.py'
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Comprehensive demo completed successfully!")
            print("📊 Results saved to 'demo_results' folder")
            
            # List generated files
            demo_results_dir = Path("demo_results")
            if demo_results_dir.exists():
                print("📁 Generated files:")
                for file in demo_results_dir.glob("*"):
                    print(f"   - {file.name}")
            
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
        # Start the dashboard
        process = subprocess.Popen([
            sys.executable, "app.py"
        ])
        
        # Wait a moment for the server to start
        time.sleep(3)
        
        print("✅ Dashboard started successfully!")
        print("🌐 Main Dashboard: http://localhost:5000")
        print("🎓 Research Demo: http://localhost:5000/research")
        
        return process
        
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        return None

def print_docker_instructions():
    """Print Docker-specific instructions"""
    print("\n" + "=" * 70)
    print("🐳 DOCKER DEMONSTRATION INSTRUCTIONS")
    print("=" * 70)
    print()
    print("1. 🎯 Access Points:")
    print("   - Main Dashboard: http://localhost:5000")
    print("   - Research Demo: http://localhost:5000/research")
    print("   - Prometheus: http://localhost:9090")
    print("   - Grafana: http://localhost:3000 (if enabled)")
    print()
    print("2. 🔧 Docker Services Running:")
    print("   - CA Service: Certificate Authority")
    print("   - Dashboard: Enhanced with research features")
    print("   - Server: Federated Learning Server")
    print("   - Clients: 10 FL clients (1-10)")
    print("   - Research Demo: Comprehensive analysis")
    print("   - Monitoring: Prometheus + Grafana")
    print("   - Redis: Caching and session management")
    print("   - Nginx: Reverse proxy")
    print()
    print("3. 📊 Research Features Available:")
    print("   - Interactive attack detection demo")
    print("   - Multiple attack types support")
    print("   - Privacy level analysis")
    print("   - Real-time visualization")
    print("   - Comprehensive evaluation")
    print("   - Results export")
    print()
    print("4. 🎓 Presentation Tips:")
    print("   - Use the research demo page for live demonstration")
    print("   - Show different attack scenarios")
    print("   - Demonstrate privacy-detection trade-off")
    print("   - Export results for documentation")
    print()
    print("5. 📁 Generated Files:")
    print("   - demo_results/: Analysis results and visualizations")
    print("   - research_results_*.json: Exported data")
    print("   - presentation_guide.md: Detailed instructions")
    print()
    print("6. 🐳 Docker Commands:")
    print("   - View logs: docker-compose logs -f dashboard")
    print("   - Stop system: docker-compose down")
    print("   - Restart: docker-compose restart dashboard")
    print("   - Access container: docker-compose exec dashboard bash")
    print()
    print("=" * 70)
    print("🎉 Ready for your final year project presentation!")
    print("=" * 70)

def main():
    """Main function"""
    print_banner()
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please check the setup.")
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
    
    # Print instructions
    print_docker_instructions()
    
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

