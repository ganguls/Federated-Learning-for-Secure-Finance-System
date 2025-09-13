#!/usr/bin/env python3
"""
Simple demo test dashboard - focused on demo data functionality
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import random
import time

app = Flask(__name__)
CORS(app)

# Demo data storage
demo_data = []

def generate_demo_data():
    """Generate demo data for 10 clients"""
    global demo_data
    demo_data = []
    
    for client_id in range(1, 11):
        # Generate realistic metrics
        base_accuracy = 0.75 + (client_id % 4) * 0.05
        base_loss = 0.4 - (client_id % 4) * 0.05
        
        # Add some random variation
        accuracy = base_accuracy + random.uniform(-0.02, 0.02)
        loss = base_loss + random.uniform(-0.02, 0.02)
        
        # Ensure values are within reasonable bounds
        accuracy = max(0.5, min(0.95, accuracy))
        loss = max(0.1, min(0.8, loss))
        
        demo_data.append({
            'client_id': client_id,
            'is_malicious': False,  # All normal for now
            'status': 'Normal',
            'accuracy': round(accuracy, 4),
            'loss': round(loss, 4),
            'attack_type': 'None',
            'demo_port': 8081 + client_id
        })

@app.route('/')
def index():
    """Main page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Demo Test Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
            .demo-section { background: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
            .controls { margin-bottom: 20px; }
            button { padding: 10px 20px; margin: 5px; background: #3498db; color: white; border: none; border-radius: 3px; cursor: pointer; }
            button:hover { background: #2980b9; }
            .success { color: #27ae60; }
            .error { color: #e74c3c; }
            table { width: 100%; border-collapse: collapse; margin-top: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .status-normal { color: #27ae60; font-weight: bold; }
            .status-malicious { color: #e74c3c; font-weight: bold; }
            .summary { display: flex; gap: 20px; margin-bottom: 20px; }
            .summary-item { background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎯 Demo Data Test Dashboard</h1>
                <p>Testing demo data functionality for FL System</p>
            </div>
            
            <div class="demo-section">
                <h2>Demo Controls</h2>
                <div class="controls">
                    <button onclick="refreshDemoData()">🔄 Refresh Demo Data</button>
                    <button onclick="simulateAttack()">🎭 Simulate Attack</button>
                    <button onclick="resetAll()">🛡️ Reset All</button>
                </div>
                
                <div class="summary">
                    <div class="summary-item">
                        <strong>Total Clients:</strong> <span id="total-clients">0</span>
                    </div>
                    <div class="summary-item">
                        <strong>Malicious Clients:</strong> <span id="malicious-count">0</span>
                    </div>
                    <div class="summary-item">
                        <strong>Last Update:</strong> <span id="last-update">Never</span>
                    </div>
                </div>
                
                <div id="status-message"></div>
                
                <table>
                    <thead>
                        <tr>
                            <th>Client ID</th>
                            <th>Status</th>
                            <th>Accuracy</th>
                            <th>Loss</th>
                            <th>Attack Type</th>
                            <th>Port</th>
                        </tr>
                    </thead>
                    <tbody id="demo-table-body">
                        <!-- Demo data will be populated here -->
                    </tbody>
                </table>
            </div>
        </div>

        <script>
            let demoData = [];
            
            function showMessage(message, type = 'success') {
                const statusDiv = document.getElementById('status-message');
                statusDiv.innerHTML = `<div class="${type}">${message}</div>`;
                setTimeout(() => {
                    statusDiv.innerHTML = '';
                }, 3000);
            }
            
            async function refreshDemoData() {
                try {
                    showMessage('Refreshing demo data...', 'success');
                    
                    // Get security status
                    const securityResponse = await fetch('/api/security/status');
                    const securityData = await securityResponse.json();
                    
                    // Get client metrics
                    const clientsResponse = await fetch('/api/metrics/clients');
                    const clientsData = await clientsResponse.json();
                    
                    // Create demo data exactly like the original frontend
                    const activeAttacks = securityData.simulator_status?.active_attacks || {};
                    const clients = [];
                    
                    for (let clientId = 1; clientId <= 10; clientId++) {
                        const clientIdStr = clientId.toString();
                        const isMalicious = activeAttacks[clientIdStr] ? true : false;
                        const attackType = activeAttacks[clientIdStr] || 'None';
                        
                        // Get client metrics if available
                        const clientInfo = clientsData[clientIdStr] || {};
                        const clientMetrics = clientInfo.metrics || {};
                        
                        // Default values
                        let accuracy = clientMetrics.accuracy || (0.8 + (clientId % 3) * 0.05);
                        let loss = clientMetrics.loss || (0.3 - (clientId % 3) * 0.05);
                        
                        // Adjust for malicious clients
                        if (isMalicious) {
                            accuracy = Math.max(0.1, accuracy - 0.5);
                            loss = Math.min(2.0, loss + 0.7);
                        }
                        
                        clients.push({
                            client_id: clientId,
                            is_malicious: isMalicious,
                            status: isMalicious ? 'Malicious' : 'Normal',
                            accuracy: Math.round(accuracy * 10000) / 10000,
                            loss: Math.round(loss * 10000) / 10000,
                            attack_type: attackType,
                            demo_port: 8081 + clientId
                        });
                    }
                    
                    demoData = clients;
                    updateDemoTable();
                    updateSummary();
                    showMessage('Demo data refreshed successfully!', 'success');
                    
                } catch (error) {
                    console.error('Error refreshing demo data:', error);
                    showMessage('Failed to refresh demo data: ' + error.message, 'error');
                }
            }
            
            function updateDemoTable() {
                const tbody = document.getElementById('demo-table-body');
                tbody.innerHTML = '';
                
                demoData.forEach(client => {
                    const row = document.createElement('tr');
                    
                    const statusClass = client.is_malicious ? 'status-malicious' : 'status-normal';
                    const statusText = client.is_malicious ? 'Malicious' : 'Normal';
                    
                    row.innerHTML = `
                        <td>${client.client_id}</td>
                        <td><span class="${statusClass}">${statusText}</span></td>
                        <td>${client.accuracy}</td>
                        <td>${client.loss}</td>
                        <td>${client.attack_type}</td>
                        <td>${client.demo_port}</td>
                    `;
                    
                    tbody.appendChild(row);
                });
            }
            
            function updateSummary() {
                const maliciousCount = demoData.filter(client => client.is_malicious).length;
                document.getElementById('malicious-count').textContent = maliciousCount;
                document.getElementById('total-clients').textContent = demoData.length;
                document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
            }
            
            async function simulateAttack() {
                try {
                    showMessage('Simulating attack...', 'success');
                    
                    // Simulate attack on random client
                    const randomClient = Math.floor(Math.random() * 10) + 1;
                    const response = await fetch(`/api/demo/toggle/${randomClient}`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ attack_type: 'random' })
                    });
                    
                    if (response.ok) {
                        showMessage(`Attack simulated on client ${randomClient}`, 'success');
                        refreshDemoData();
                    } else {
                        showMessage('Failed to simulate attack', 'error');
                    }
                } catch (error) {
                    showMessage('Error simulating attack: ' + error.message, 'error');
                }
            }
            
            async function resetAll() {
                try {
                    showMessage('Resetting all clients...', 'success');
                    
                    const response = await fetch('/api/demo/reset_all', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' }
                    });
                    
                    if (response.ok) {
                        showMessage('All clients reset to normal', 'success');
                        refreshDemoData();
                    } else {
                        showMessage('Failed to reset clients', 'error');
                    }
                } catch (error) {
                    showMessage('Error resetting clients: ' + error.message, 'error');
                }
            }
            
            // Auto-refresh on page load
            window.onload = function() {
                refreshDemoData();
            };
        </script>
    </body>
    </html>
    """

@app.route('/api/security/status')
def get_security_status():
    """Get security status"""
    return jsonify({
        'security_status': 'NORMAL',
        'attack_detection_enabled': True,
        'malicious_clients': [],
        'simulator_status': {
            'active_attacks': {},
            'total_attacks': 0,
            'blocked_clients': []
        },
        'message': 'Security system active'
    })

@app.route('/api/metrics/clients')
def get_client_metrics():
    """Get client metrics"""
    client_metrics = {}
    
    for client_id in range(1, 11):
        # Generate realistic metrics with some variation
        base_accuracy = 0.75 + (client_id % 4) * 0.05
        base_loss = 0.4 - (client_id % 4) * 0.05
        
        # Add some random variation
        accuracy = base_accuracy + random.uniform(-0.02, 0.02)
        loss = base_loss + random.uniform(-0.02, 0.02)
        
        # Ensure values are within reasonable bounds
        accuracy = max(0.5, min(0.95, accuracy))
        loss = max(0.1, min(0.8, loss))
        
        client_metrics[str(client_id)] = {
            'status': 'running',
            'uptime': f'{random.randint(1, 60)}m {random.randint(0, 59)}s',
            'metrics': {
                'accuracy': round(accuracy, 4),
                'loss': round(loss, 4),
                'precision': round(accuracy + random.uniform(-0.02, 0.02), 4),
                'recall': round(accuracy + random.uniform(-0.02, 0.02), 4),
                'f1_score': round(accuracy + random.uniform(-0.02, 0.02), 4)
            }
        }
    
    return jsonify(client_metrics)

@app.route('/api/demo/toggle/<int:client_id>', methods=['POST'])
def toggle_client_malicious(client_id):
    """Toggle malicious status for a specific client"""
    return jsonify({
        'success': True,
        'message': f'Client {client_id} status toggled',
        'client_id': client_id
    })

@app.route('/api/demo/reset_all', methods=['POST'])
def reset_all_clients():
    """Reset all clients to normal status"""
    return jsonify({
        'success': True,
        'message': 'All clients reset to normal status'
    })

if __name__ == '__main__':
    print("🚀 Starting Demo Test Dashboard...")
    print("🌐 Dashboard URL: http://localhost:5000")
    print("📊 Demo data functionality will be tested")
    print("Press Ctrl+C to stop")
    
    app.run(host='0.0.0.0', port=5000, debug=True)

