import json
import os
from datasets.soulforge.extract_dependenty import EnhancedLeanStructureExtractor
#LeanStructureExtractor
import networkx as nx

class LeanHTMLVisualizer(EnhancedLeanStructureExtractor):
    def generate_hierarchy_data(self):
        """Generate hierarchical JSON data structure for visualization."""
        if not self.hierarchy_graph:
            self.build_hierarchy()
        
        # Find root nodes (those without dependencies)
        roots = [node for node in self.hierarchy_graph.nodes if self.hierarchy_graph.out_degree(node) == 0]
        
        # If no true roots, use nodes with highest in-degree as virtual roots
        if not roots:
            in_degrees = sorted([(n, self.hierarchy_graph.in_degree(n)) for n in self.hierarchy_graph.nodes], 
                                key=lambda x: x[1], reverse=True)
            # Take top 5 as virtual roots
            roots = [n for n, _ in in_degrees[:5]]
        
        # For each root, build its subtree
        forest = []
        processed_nodes = set()
        
        def build_subtree(node):
            if node in processed_nodes:
                # Return a reference node to avoid cycles
                return {"name": node, "isReference": True}
            
            processed_nodes.add(node)
            children = list(self.reverse_deps.get(node, []))
            info = self.structures.get(node, {})
            
            # Create node data
            node_data = {
                "name": node,
                "fullName": info.get('full_name', node),
                "namespace": info.get('namespace', ''),
                "extends": info.get('extends', []),
                "size": len(children) + 1,  # Size based on number of children
                "children": [build_subtree(child) for child in children]
            }
            
            return node_data
        
        for root in roots:
            tree = build_subtree(root)
            forest.append(tree)
        
        # If we have multiple roots, create a virtual root
        if len(forest) > 1:
            root_data = {
                "name": "Root Structures",
                "isVirtual": True,
                "children": forest
            }
            return root_data
        elif forest:
            return forest[0]
        else:
            return {"name": "No structures found", "children": []}
    
    def generate_html(self, output_file="lean_structures_hierarchy.html"):
        """Generate an HTML file with interactive D3.js visualization."""
        # Generate hierarchical data
        hierarchy_data = self.generate_hierarchy_data()
        
        # Read the HTML template
        html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lean Structure Hierarchy</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }
        
        #container {
            display: flex;
            height: 100vh;
        }
        
        #sidebar {
            width: 300px;
            background-color: #f0f0f0;
            padding: 20px;
            overflow-y: auto;
            box-shadow: 2px 0 5px rgba(0,0,0,0.1);
        }
        
        #visualization {
            flex-grow: 1;
            overflow: auto;
            padding: 20px;
        }
        
        #search {
            margin-bottom: 15px;
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        
        #info-panel {
            margin-top: 20px;
            border-top: 1px solid #ddd;
            padding-top: 15px;
        }
        
        h1 {
            font-size: 1.8em;
            margin-top: 0;
            color: #333;
        }
        
        h2 {
            font-size: 1.2em;
            margin-top: 10px;
            color: #555;
        }
        
        p {
            margin: 5px 0;
        }
        
        .node circle {
            fill: #E8F8F5;
            stroke: steelblue;
            stroke-width: 1.5px;
        }
        
        .node.virtual circle {
            fill: #f0f0f0;
            stroke: #999;
        }
        
        .node.reference circle {
            fill: #FFE6CC;
            stroke: #D79B00;
        }
        
        .node text {
            font: 12px sans-serif;
            dominant-baseline: central;
        }
        
        .node.selected circle {
            fill: #FFCCBC;
            stroke: #E64A19;
            stroke-width: 2.5px;
        }
        
        .link {
            fill: none;
            stroke: #ccc;
            stroke-width: 1.5px;
        }
        
        .controls {
            position: absolute;
            top: 10px;
            right: 10px;
            background: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.2);
        }
        
        button {
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 5px 10px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 14px;
            margin: 2px;
            cursor: pointer;
            border-radius: 3px;
        }
        
        button:hover {
            background-color: #45a049;
        }
        
        .stat {
            display: inline-block;
            margin-right: 15px;
            font-weight: bold;
        }
        
        .search-result {
            padding: 5px;
            margin: 2px 0;
            cursor: pointer;
            border-radius: 3px;
        }
        
        .search-result:hover {
            background-color: #e0e0e0;
        }
        
        .highlight {
            background-color: yellow;
        }
    </style>
</head>
<body>
    <div id="container">
        <div id="sidebar">
            <h1>Lean Structure Hierarchy</h1>
            <input type="text" id="search" placeholder="Search structures...">
            <div id="search-results"></div>
            <div id="info-panel">
                <h2>Structure Information</h2>
                <p>Click on a node to view details</p>
            </div>
            <div id="stats">
                <h2>Statistics</h2>
                <div>
                    <span class="stat">Total: <span id="total-count">0</span></span>
                    <span class="stat">Levels: <span id="level-count">0</span></span>
                </div>
            </div>
        </div>
        <div id="visualization">
            <div class="controls">
                <button id="zoom-in">+</button>
                <button id="zoom-out">-</button>
                <button id="reset">Reset</button>
                <button id="expand-all">Expand All</button>
                <button id="collapse-all">Collapse All</button>
            </div>
        </div>
    </div>

    <script>
        // Tree visualization data
        const treeData = %s;
        
        // Keep track of all nodes for search
        let allNodes = [];
        
        // Set dimensions and margins
        const margin = {top: 50, right: 90, bottom: 50, left: 90};
        const width = 1200 - margin.left - margin.right;
        const height = 800 - margin.top - margin.bottom;
        
        // Create SVG container
        const svg = d3.select("#visualization").append("svg")
            .attr("width", "100%%")
            .attr("height", "100%%")
            .attr("viewBox", [-margin.left, -margin.top, width + margin.left + margin.right, height + margin.top + margin.bottom])
            .call(d3.zoom().on("zoom", (event) => {
                g.attr("transform", event.transform);
            }))
            .append("g");
        
        const g = svg.append("g");
        
        // Create tree layout
        const tree = d3.tree()
            .size([height, width]);
        
        // Compute node positions
        const root = d3.hierarchy(treeData);
        root.x0 = height / 2;
        root.y0 = 0;
        
        // Collapse nodes initially
        root.children.forEach(collapse);
        
        update(root);
        
        // Calculate statistics
        document.getElementById('total-count').textContent = allNodes.length;
        document.getElementById('level-count').textContent = root.height;
        
        // Function to collapse node
        function collapse(d) {
            if (d.children) {
                d._children = d.children;
                d._children.forEach(collapse);
                d.children = null;
            }
        }
        
        // Function to expand node
        function expand(d) {
            if (d._children) {
                d.children = d._children;
                d.children.forEach(expand);
                d._children = null;
            }
        }
        
        // Toggle node expansion/collapse
        function toggle(d) {
            if (d.children) {
                d._children = d.children;
                d.children = null;
            } else if (d._children) {
                d.children = d._children;
                d._children = null;
            }
            update(d);
        }
        
        // Update the tree visualization
        function update(source) {
            // Compute the new tree layout
            const treeData = tree(root);
            
            // Get nodes and links
            const nodes = treeData.descendants();
            const links = treeData.links();
            
            // Update all nodes list for search
            allNodes = nodes.map(d => d.data);
            
            // Normalize for fixed-depth
            nodes.forEach(d => {
                d.y = d.depth * 180;
            });
            
            // ****************** Nodes section ***************************
            
            // Update the nodes
            const node = g.selectAll('.node')
                .data(nodes, d => d.id || (d.id = ++i));
            
            // Enter new nodes at the parent's previous position
            const nodeEnter = node.enter().append('g')
                .attr('class', d => {
                    let className = 'node';
                    if (d.data.isReference) className += ' reference';
                    if (d.data.isVirtual) className += ' virtual';
                    return className;
                })
                .attr('transform', d => `translate(\${d.y},\${d.x})`)
                .on('click', (event, d) => {
                    // Clear previous selection
                    d3.selectAll('.node').classed('selected', false);
                    
                    // Select this node
                    d3.select(event.currentTarget).classed('selected', true);
                    
                    // Show info in sidebar
                    showInfo(d.data);
                    
                    // Toggle expansion
                    toggle(d);
                });
            
            // Add Circle for the nodes
            nodeEnter.append('circle')
                .attr('r', d => Math.min(10, 5 + Math.sqrt(d.data.size || 1)))
                .style('fill', d => d._children ? '#E8F8F5' : '#fff');
            
            // Add labels for the nodes
            nodeEnter.append('text')
                .attr('dy', '.35em')
                .attr('x', d => d.children || d._children ? -13 : 13)
                .attr('text-anchor', d => d.children || d._children ? 'end' : 'start')
                .text(d => d.data.name);
            
            // UPDATE
            const nodeUpdate = nodeEnter.merge(node);
            
            // Transition to the proper position for the nodes
            nodeUpdate.transition()
                .duration(500)
                .attr('transform', d => `translate(\${d.y},\${d.x})`);
            
            // Update node attributes and style
            nodeUpdate.select('circle')
                .attr('r', d => Math.min(10, 5 + Math.sqrt(d.data.size || 1)))
                .style('fill', d => d._children ? '#E8F8F5' : '#fff');
            
            // Remove any exiting nodes
            const nodeExit = node.exit().transition()
                .duration(500)
                .attr('transform', d => `translate(\${source.y},\${source.x})`)
                .remove();
            
            // Reduce the node circles size to 0
            nodeExit.select('circle')
                .attr('r', 0);
            
            // ****************** links section ***************************
            
            // Update the links
            const link = g.selectAll('path.link')
                .data(links, d => d.target.id);
            
            // Enter any new links at the parent's previous position
            const linkEnter = link.enter().insert('path', 'g')
                .attr('class', 'link')
                .attr('d', d => {
                    const o = {x: source.x0, y: source.y0};
                    return diagonal(o, o);
                });
            
            // UPDATE
            const linkUpdate = linkEnter.merge(link);
            
            // Transition back to the parent element position
            linkUpdate.transition()
                .duration(500)
                .attr('d', d => diagonal(d.source, d.target));
            
            // Remove any exiting links
            link.exit().transition()
                .duration(500)
                .attr('d', d => {
                    const o = {x: source.x, y: source.y};
                    return diagonal(o, o);
                })
                .remove();
            
            // Store the old positions for transition
            nodes.forEach(d => {
                d.x0 = d.x;
                d.y0 = d.y;
            });
        }
        
        // Creates a curved (diagonal) path from parent to the child nodes
        function diagonal(s, d) {
            return `M \${s.y} \${s.x}
                    C \${(s.y + d.y) / 2} \${s.x},
                      \${(s.y + d.y) / 2} \${d.x},
                      \${d.y} \${d.x}`;
        }
        
        // Show node information in the sidebar
        function showInfo(nodeData) {
            const infoPanel = document.getElementById('info-panel');
            
            // Don't show details for virtual nodes
            if (nodeData.isVirtual) {
                infoPanel.innerHTML = `
                    <h2>\${nodeData.name}</h2>
                    <p>Virtual node grouping multiple root structures</p>
                `;
                return;
            }
            
            // Handle reference nodes
            if (nodeData.isReference) {
                infoPanel.innerHTML = `
                    <h2>\${nodeData.name}</h2>
                    <p>Reference to structure (used elsewhere in hierarchy)</p>
                `;
                return;
            }
            
            // Regular node information
            let html = `
                <h2>\${nodeData.name}</h2>
            `;
            
            if (nodeData.namespace) {
                html += `<p><strong>Namespace:</strong> \${nodeData.namespace}</p>`;
            }
            
            if (nodeData.fullName) {
                html += `<p><strong>Full name:</strong> \${nodeData.fullName}</p>`;
            }
            
            if (nodeData.extends && nodeData.extends.length > 0) {
                html += `<p><strong>Extends:</strong> \${nodeData.extends.join(', ')}</p>`;
            }
            
            // Show children
            if ((nodeData.children && nodeData.children.length > 0) || 
                (nodeData._children && nodeData._children.length > 0)) {
                
                const children = nodeData.children || nodeData._children;
                html += `<p><strong>Extended by:</strong> \${children.map(c => c.name).join(', ')}</p>`;
            }
            
            infoPanel.innerHTML = html;
        }
        
        // Search functionality
        const searchInput = document.getElementById('search');
        const searchResults = document.getElementById('search-results');
        
        searchInput.addEventListener('input', function() {
            const query = this.value.toLowerCase();
            if (query.length < 2) {
                searchResults.innerHTML = '';
                return;
            }
            
            // Find matching nodes
            const matches = allNodes.filter(node => 
                node.name.toLowerCase().includes(query) || 
                (node.fullName && node.fullName.toLowerCase().includes(query))
            ).slice(0, 20); // Limit to 20 results
            
            // Display results
            searchResults.innerHTML = matches.map(node => 
                `<div class="search-result" data-name="\${node.name}">
                    \${highlightText(node.name, query)}
                    \${node.namespace ? `<small> (\${node.namespace})</small>` : ''}
                </div>`
            ).join('');
            
            // Add click listeners
            searchResults.querySelectorAll('.search-result').forEach(result => {
                result.addEventListener('click', function() {
                    const name = this.dataset.name;
                    
                    // Find the node in the tree
                    const node = findNodeByName(root, name);
                    if (node) {
                        // Expand path to node
                        expandToNode(root, node);
                        
                        // Update the tree
                        update(root);
                        
                        // Select the node
                        setTimeout(() => {
                            const nodeElement = document.querySelector(`g.node text:contains('\${name}')`);
                            if (nodeElement) {
                                nodeElement.parentNode.dispatchEvent(new Event('click'));
                                
                                // Scroll to the node
                                nodeElement.scrollIntoView({
                                    behavior: 'smooth',
                                    block: 'center'
                                });
                            }
                        }, 600);
                    }
                });
            });
        });
        
        // Helper to find a node by name
        function findNodeByName(root, name) {
            if (root.data.name === name) {
                return root;
            }
            
            if (root.children) {
                for (const child of root.children) {
                    const found = findNodeByName(child, name);
                    if (found) return found;
                }
            }
            
            if (root._children) {
                for (const child of root._children) {
                    const found = findNodeByName(child, name);
                    if (found) return found;
                }
            }
            
            return null;
        }
        
        // Expand path to a specific node
        function expandToNode(root, target) {
            if (root === target) {
                return true;
            }
            
            if (root._children) {
                root.children = root._children;
                root._children = null;
                
                for (const child of root.children) {
                    if (expandToNode(child, target)) {
                        return true;
                    }
                }
                
                // If not found in any children, collapse again
                root._children = root.children;
                root.children = null;
            }
            
            return false;
        }
        
        // Helper to highlight search matches
        function highlightText(text, query) {
            const index = text.toLowerCase().indexOf(query.toLowerCase());
            if (index >= 0) {
                return text.substring(0, index) + 
                    `<span class="highlight">\${text.substring(index, index + query.length)}</span>` + 
                    text.substring(index + query.length);
            }
            return text;
        }
        
        // Custom contains selector for case-insensitive text search
        jQuery.expr[':'].contains = function(a, i, m) {
            return jQuery(a).text().toUpperCase().indexOf(m[3].toUpperCase()) >= 0;
        };
        
        // Control buttons
        document.getElementById('zoom-in').addEventListener('click', function() {
            const zoom = d3.zoom().on("zoom", (event) => {
                g.attr("transform", event.transform);
            });
            
            svg.transition().call(zoom.scaleBy, 1.3);
        });
        
        document.getElementById('zoom-out').addEventListener('click', function() {
            const zoom = d3.zoom().on("zoom", (event) => {
                g.attr("transform", event.transform);
            });
            
            svg.transition().call(zoom.scaleBy, 0.7);
        });
        
        document.getElementById('reset').addEventListener('click', function() {
            const zoom = d3.zoom().on("zoom", (event) => {
                g.attr("transform", event.transform);
            });
            
            svg.transition().call(zoom.transform, d3.zoomIdentity);
            update(root);
        });
        
        document.getElementById('expand-all').addEventListener('click', function() {
            expand(root);
            update(root);
        });
        
        document.getElementById('collapse-all').addEventListener('click', function() {
            root.children.forEach(collapse);
            update(root);
        });
        
        // Fix selection for D3 v7
        let i = 0;
    </script>
</body>
</html>
""" % json.dumps(hierarchy_data)
        
        # Write the HTML file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_template)
        
        print(f"Interactive HTML visualization saved to {output_file}")
        return output_file

def main(jsonl_file, output_file="lean_structures_hierarchy.html"):
    visualizer = LeanHTMLVisualizer()
    visualizer.parse_file(jsonl_file)
    visualizer.build_hierarchy()
    visualizer.print_hierarchy()
    visualizer.generate_html(output_file)
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize Lean structure hierarchy")
    parser.add_argument("--jsonl_file", default = "data/leandojo_benchmark_4/corpus.jsonl", help="Path to the Lean corpus JSONL file")

    args = parser.parse_args()
    main(args.jsonl_file)