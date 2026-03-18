import {
  WebSocketGateway,
  WebSocketServer,
  OnGatewayConnection,
  OnGatewayDisconnect,
  SubscribeMessage,
  MessageBody,
  ConnectedSocket,
} from '@nestjs/websockets';
import { Server, Socket } from 'socket.io';
import { Logger } from '@nestjs/common';

interface LiveUpdatePayload {
  timestamp: string;
  count: number;
  matches: unknown[];
}

/**
 * WebSocket gateway that pushes live score updates to connected clients.
 *
 * Events emitted to clients:
 *   - live:update        — full list of all currently live matches
 *   - match:update       — single match update (for targeted subscriptions)
 *
 * Events received from clients:
 *   - subscribe:match    — { matchId: number } — subscribe to a specific match
 *   - unsubscribe:match  — { matchId: number } — unsubscribe from a specific match
 */
@WebSocketGateway({
  cors: { origin: process.env.CORS_ORIGIN || '*' },
  namespace: '/live',
  transports: ['websocket', 'polling'],
})
export class LiveScoresGateway
  implements OnGatewayConnection, OnGatewayDisconnect
{
  @WebSocketServer()
  private readonly server!: Server;

  private readonly logger = new Logger(LiveScoresGateway.name);
  private connectedClients = 0;

  handleConnection(client: Socket) {
    this.connectedClients++;
    this.logger.debug(
      `Client connected: ${client.id} (total: ${this.connectedClients})`,
    );
    // Send the connection confirmation
    client.emit('connected', {
      message: 'Connected to live scores feed',
      timestamp: new Date().toISOString(),
    });
  }

  handleDisconnect(client: Socket) {
    this.connectedClients--;
    this.logger.debug(
      `Client disconnected: ${client.id} (total: ${this.connectedClients})`,
    );
  }

  // ── Subscriptions ──────────────────────────────────────────────────────

  @SubscribeMessage('subscribe:match')
  handleSubscribeMatch(
    @ConnectedSocket() client: Socket,
    @MessageBody() data: { matchId: number },
  ) {
    const room = `match:${data.matchId}`;
    void client.join(room);
    this.logger.debug(`Client ${client.id} subscribed to ${room}`);
    return { subscribed: true, room };
  }

  @SubscribeMessage('unsubscribe:match')
  handleUnsubscribeMatch(
    @ConnectedSocket() client: Socket,
    @MessageBody() data: { matchId: number },
  ) {
    const room = `match:${data.matchId}`;
    void client.leave(room);
    return { unsubscribed: true, room };
  }

  @SubscribeMessage('ping')
  handlePing() {
    return { pong: true, timestamp: new Date().toISOString() };
  }

  // ── Broadcast methods (called by LiveUpdatesProcessor) ────────────────

  /**
   * Broadcast all live matches to every connected client.
   * Also emits match:update to clients subscribed to specific match rooms.
   */
  broadcastLiveUpdate(matches: unknown[]): void {
    if (this.connectedClients === 0) return;

    const payload: LiveUpdatePayload = {
      timestamp: new Date().toISOString(),
      count: (matches as { id: number }[]).length,
      matches,
    };

    this.server.emit('live:update', payload);

    // Also push targeted updates to per-match subscribers
    for (const match of matches as { id: number }[]) {
      this.server.to(`match:${match.id}`).emit('match:update', match);
    }

    this.logger.debug(
      `Broadcast live:update — ${payload.count} matches to ${this.connectedClients} clients`,
    );
  }

  getConnectedClientsCount(): number {
    return this.connectedClients;
  }
}
