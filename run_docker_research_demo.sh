#!/bin/bash

# Federated Learning Attack Detection - Docker Research Demo
# Final Year Project Presentation System

echo "================================================================"
echo "🎓 FEDERATED LEARNING ATTACK DETECTION - DOCKER DEMO"
echo "================================================================"
echo "Final Year Project Presentation System"
echo "Data Poisoning Prevention in Federated Learning"
echo "================================================================"
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if Docker is running
echo "🔍 Checking Docker environment..."
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi
print_status "Docker is running"

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose."
    exit 1
fi
print_status "Docker Compose is available"

# Clean up any existing containers
echo
echo "🧹 Cleaning up existing containers..."
docker-compose down -v 2>/dev/null || true
print_status "Cleanup completed"

# Build the images
echo
echo "🔨 Building Docker images..."
print_info "This may take a few minutes on first run..."

if docker-compose build --no-cache; then
    print_status "Images built successfully"
else
    print_error "Failed to build images"
    exit 1
fi

# Start the services
echo
echo "🚀 Starting Federated Learning system..."
print_info "Starting all services (CA, Server, Clients, Dashboard, Research Demo)..."

if docker-compose up -d; then
    print_status "All services started successfully"
else
    print_error "Failed to start services"
    exit 1
fi

# Wait for services to be ready
echo
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check service status
echo
echo "🔍 Checking service status..."
services=("ca" "server" "dashboard" "client1" "client2" "client3" "client4" "client5" "client6" "client7" "client8" "client9" "client10" "research-demo")

for service in "${services[@]}"; do
    if docker-compose ps | grep -q "$service.*Up"; then
        print_status "$service is running"
    else
        print_warning "$service may not be running properly"
    fi
done

# Run the research demo
echo
echo "🎓 Running research demonstration..."
print_info "Executing comprehensive analysis..."

if docker-compose exec -T research-demo python demo_presentation.py; then
    print_status "Research demo completed successfully"
else
    print_warning "Research demo had issues, but system is still running"
fi

# Display access information
echo
echo "================================================================"
echo "🌐 ACCESS INFORMATION"
echo "================================================================"
echo
echo "📊 Main Dashboard:"
echo "   URL: http://localhost:5000"
echo "   Features: System monitoring, client management, training metrics"
echo
echo "🎓 Research Demo:"
echo "   URL: http://localhost:5000/research"
echo "   Features: Interactive attack detection, privacy analysis, visualization"
echo
echo "📈 Monitoring:"
echo "   Prometheus: http://localhost:9090"
echo "   Grafana: http://localhost:3000 (if enabled)"
echo
echo "🔧 Management Commands:"
echo "   View logs: docker-compose logs -f [service_name]"
echo "   Stop system: docker-compose down"
echo "   Restart: docker-compose restart [service_name]"
echo "   Access container: docker-compose exec [service_name] bash"
echo
echo "📁 Generated Files:"
echo "   Results: Check 'demo_results' volume"
echo "   Logs: Check service-specific log volumes"
echo

# Display research demo results
echo "================================================================"
echo "📊 RESEARCH DEMO RESULTS"
echo "================================================================"
echo

# Check if demo results exist
if docker-compose exec -T research-demo ls -la demo_results/ 2>/dev/null | grep -q "\.png\|\.json\|\.md"; then
    print_status "Demo results generated successfully"
    echo
    echo "Generated files:"
    docker-compose exec -T research-demo ls -la demo_results/ | grep -E "\.(png|json|md)$" | awk '{print "   - " $9}'
else
    print_warning "Demo results not found. Check research-demo container logs."
fi

echo
echo "================================================================"
echo "🎉 SYSTEM READY FOR PRESENTATION!"
echo "================================================================"
echo
echo "🎯 Next Steps:"
echo "1. Open http://localhost:5000/research in your browser"
echo "2. Follow the presentation guide in presentation_guide.md"
echo "3. Demonstrate different attack scenarios"
echo "4. Show privacy-detection trade-off analysis"
echo "5. Export results for documentation"
echo
echo "📖 For detailed instructions, check:"
echo "   - presentation_guide.md"
echo "   - RESEARCH_PROJECT_SUMMARY.md"
echo
echo "🛑 To stop the system: docker-compose down"
echo "================================================================"

# Keep the script running and show logs
echo
echo "📋 Showing dashboard logs (Press Ctrl+C to stop):"
echo "================================================================"
docker-compose logs -f dashboard
