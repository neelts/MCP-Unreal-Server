import asyncio
import logging
import time
from typing import Dict, List, Optional, AsyncIterator, Any
from contextlib import asynccontextmanager
from collections import deque

from mcp.server.models import InitializationOptions

# Configure logging
logging.basicConfig(level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
_logger = logging.getLogger("UnrealMCPServer")

# Add this line to lower the log level of mcp.server.lowlevel.server
# Initially set to Error level, causing warning messages in Cline, so lower the log level
logging.getLogger("mcp.server.lowlevel.server").setLevel(logging.WARNING)

# Configure file handler with more concise format for frequent operations
file_handler = logging.FileHandler('mcp_unreal.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname).1s - %(message)s', '%H:%M:%S'))
_logger.addHandler(file_handler)

# Configure console handler with detailed format
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
_logger.addHandler(console_handler)

import mcp.types as types
from mcp.server import NotificationOptions, Server
from pydantic import AnyUrl
import mcp.server.stdio

from .mcp_server_unreal.remote_execution import RemoteExecution, RemoteExecutionConfig,MODE_EXEC_FILE,MODE_EXEC_STATEMENT,MODE_EVAL_STATEMENT

# Global connection variable
_unreal_connection: Optional[RemoteExecution] = None
_node_monitor_task: Optional[asyncio.Task] = None

def get_unreal_connection(host: str = "239.0.0.1", port: int = 6766) -> RemoteExecution:
    """Get or create a persistent Unreal connection"""
    global _unreal_connection
    
    # If there is an existing connection, check if it is still valid
    if _unreal_connection is not None:
        try:
            nodes = _unreal_connection.remote_nodes
            return _unreal_connection
        except Exception as e:
            _logger.warning(f"Existing connection is invalid: {str(e)}")
            try:
                _unreal_connection.stop()
            except:
                pass
            _unreal_connection = None
    
    # Create a new connection
    if _unreal_connection is None:
        config = RemoteExecutionConfig()
        config.multicast_group_endpoint = (host, port)
        _unreal_connection = RemoteExecution(config)
        _unreal_connection.start()
        _logger.info("Created a new persistent Unreal connection")
    
    return _unreal_connection

class McpUnrealServer:
    def __init__(self, server_name: str, lifespan=None):
        self.server = Server(server_name, lifespan=lifespan)
        self.remote_execution = None
        self.connected_nodes: Dict[str, dict] = {}
        self._node_monitor_task = None
        self._setup_handlers()

    def _setup_handlers(self):
        @self.server.list_resources()
        async def handle_list_resources() -> list[types.Resource]:
            """List available Unreal node resources."""
            resources = []
            if self.remote_execution:
                for node in self.remote_execution.remote_nodes:
                    resources.append(
                        types.Resource(
                            uri=AnyUrl(f"unreal://{node['node_id']}"),
                            name=f"Unreal Instance: {node['node_id']}",
                            description="Unreal Engine instance",
                            mimeType="application/x-unreal",
                        )
                    )
            return resources

        @self.server.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            """List available tools."""
            return  [
                types.Tool(
                    name="execute-python",
                    description="Execute Python code in Unreal",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "code": {"type": "string"},
                            "unattended": {"type": "boolean", "default": True},
                        },
                        "required": ["code"],
                    },
                ),
            ]

        # Add resource template handler
        @self.server.list_resource_templates()
        async def handle_list_resource_templates() -> list[types.ResourceTemplate]:
            """List available resource templates."""
            return []

        @self.server.call_tool()
        async def handle_call_tool(
            name: str, arguments: dict | None
        ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
            """Handle tool execution request."""
            if name == "connect-unreal":
                return await self._handle_connect_unreal(arguments or {})
            elif name == "execute-python":
                return await self._handle_execute_python(arguments or {})
            raise ValueError(f"Unknown tool: {name}")

    async def _handle_connect_unreal(self, arguments: dict) -> list[types.TextContent]:
        """Handle Unreal connection request."""
        try:
            host = arguments.get("host", "239.0.0.1")
            port = arguments.get("port", 6766)
            _logger.info(f"Attempting to connect to Unreal: host={host}, port={port}")

            if self.remote_execution:
                self.remote_execution.stop()

            config = RemoteExecutionConfig()
            config.multicast_group_endpoint = (host, port)
            
            self.remote_execution = RemoteExecution(config)
            self.remote_execution.start()

            # Wait for nodes to be discovered
            await asyncio.sleep(2)
            nodes = self.remote_execution.remote_nodes
            
            if not nodes:
                _logger.warning("No Unreal nodes found")
                return [types.TextContent(type="text", text="No Unreal nodes found")]

            # Update the list of connected nodes
            self.connected_nodes = {node["node_id"]: node for node in nodes}
            await self.server.request_context.session.send_resource_list_changed()

            # Start the node monitoring task
            if self._node_monitor_task:
                self._node_monitor_task.cancel()
            self._node_monitor_task = asyncio.create_task(self._monitor_nodes())

            _logger.info(f"Successfully connected to Unreal, found {len(nodes)} nodes")
            _logger.info(f"Current node list: {self.connected_nodes.keys()}")
            return [types.TextContent(
                type="text",
                text=f"Successfully connected to Unreal, found {len(nodes)} nodes"
            )]
        except Exception as e:
            _logger.error(f"Failed to connect to Unreal: {str(e)}")
            return [types.TextContent(
                type="text",
                text=f"Failed to connect to Unreal: {str(e)}"
            )]

    async def _handle_execute_python(self, arguments: dict) -> list[types.TextContent]:
        """Handle Python code execution request."""
        global _unreal_connection
        
        # Ensure the connection exists and is valid
        try:
            if not _unreal_connection or not _unreal_connection.remote_nodes:
                _unreal_connection = get_unreal_connection()
                # Wait a short time to ensure the connection is established
                await asyncio.sleep(1)
                
            if not _unreal_connection or not _unreal_connection.remote_nodes:
                return [types.TextContent(type="text", text="Unable to connect to Unreal instance, please ensure Unreal is running and remote execution is enabled")]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Failed to connect to Unreal: {str(e)}")]

        code = arguments.get("code")
        if not code:
            return [types.TextContent(type="text", text="No Python code provided")]

        unattended = arguments.get("unattended", True)
        exec_mode = MODE_EXEC_STATEMENT

        try:
            # Get the first available node
            nodes = _unreal_connection.remote_nodes
            if not nodes:
                return [types.TextContent(type="text", text="No Unreal nodes found")]
            
            node_id = nodes[0]["node_id"]
            _unreal_connection.open_command_connection(node_id)
            
            result = _unreal_connection.run_command(
                code, unattended=unattended, exec_mode=exec_mode
            )
            _unreal_connection.close_command_connection()

            if not result.get("success", False):
                return [types.TextContent(
                    type="text",
                    text=f"Execution failed: {result.get('result', 'Unknown error')}"
                )]

            return [types.TextContent(
                type="text",
                text=f"Execution result:\n{result.get('result', '')}"
            )]
        except Exception as e:
            if _unreal_connection:
                try:
                    _unreal_connection.close_command_connection()
                except:
                    pass
            return [types.TextContent(
                type="text",
                text=f"Execution failed: {str(e)}"
            )]

    async def _monitor_nodes(self):
        """Asynchronous task to monitor node status."""
        while True:
            try:
                await asyncio.sleep(1)  # Check every second
                if not self.remote_execution:
                    break

                current_nodes = {node["node_id"]: node for node in self.remote_execution.remote_nodes}
                
                # Check for node changes
                if current_nodes != self.connected_nodes:
                    self.connected_nodes = current_nodes
                    await self.server.request_context.session.send_resource_list_changed()
            except asyncio.CancelledError:
                break
            except Exception as e:
                _logger.error(f"Node monitoring error: {str(e)}")

    async def close(self):
        """Close the server and all connections."""
        if self._node_monitor_task:
            self._node_monitor_task.cancel()
            try:
                await self._node_monitor_task
            except asyncio.CancelledError:
                pass

        if self.remote_execution:
            self.remote_execution.stop()

@asynccontextmanager
async def server_lifespan(server: Server) -> AsyncIterator[Dict[str, Any]]:
    """Manage server startup and shutdown lifecycle"""
    try:
        # Log server startup
        _logger.info("UnrealMCP server is starting")
        
        # Try to connect to Unreal at startup
        try:
            # This will initialize the global connection
            unreal = get_unreal_connection()
            _logger.info("Successfully connected to Unreal")
        except Exception as e:
            _logger.warning(f"Failed to connect to Unreal at startup: {str(e)}")
            _logger.warning("Please ensure the Unreal instance is running and remote execution is enabled")
        
        # Return empty context - we use global connection
        yield {}
    finally:
        # Clean up global connection on shutdown
        global _unreal_connection, _node_monitor_task
        if _node_monitor_task:
            _node_monitor_task.cancel()
            try:
                await _node_monitor_task
            except asyncio.CancelledError:
                pass
            _node_monitor_task = None
            
        if _unreal_connection:
            _logger.info("Disconnecting from Unreal")
            _unreal_connection.stop()
            _unreal_connection = None
        _logger.info("UnrealMCP server has shut down")

async def main():
    unreal_server = McpUnrealServer("mcp-server-unreal", lifespan=server_lifespan)
    try:
        # Use the server object in the instance to maintain handler registration consistency
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await unreal_server.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="mcp-server-unreal",
                    server_version="0.1.0",
                    capabilities=unreal_server.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
    finally:
        unreal_server.close()

if __name__ == "__main__":
    asyncio.run(main())