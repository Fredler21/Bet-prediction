import { NestFactory } from '@nestjs/core';
import { ValidationPipe, Logger } from '@nestjs/common';
import { DocumentBuilder, SwaggerModule } from '@nestjs/swagger';
import { IoAdapter } from '@nestjs/platform-socket.io';
import { AppModule } from './app.module';

async function bootstrap() {
  const app = await NestFactory.create(AppModule, { bufferLogs: true });

  // Global validation
  app.useGlobalPipes(
    new ValidationPipe({
      transform: true,
      whitelist: true,
      forbidNonWhitelisted: false,
    }),
  );

  // WebSocket adapter
  app.useWebSocketAdapter(new IoAdapter(app));

  // CORS (restrict origins in production via env)
  app.enableCors({
    origin: process.env.CORS_ORIGIN || '*',
    methods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE'],
  });

  // Global prefix
  app.setGlobalPrefix('api');

  // ── Health check (used by Railway / Render / Docker healthchecks) ──
  const httpAdapter = app.getHttpAdapter();
  httpAdapter.get('/api/health', (_req: unknown, res: { json: (body: unknown) => void }) => {
    res.json({ status: 'ok', timestamp: new Date().toISOString() });
  });

  // Swagger docs
  if (process.env.NODE_ENV !== 'production') {
    const config = new DocumentBuilder()
      .setTitle('Sports Collector API')
      .setDescription('Internal REST API for sports data — fixtures, standings, players, live scores')
      .setVersion('1.0')
      .addTag('matches', 'Match fixtures and live scores')
      .addTag('leagues', 'Leagues and seasons')
      .addTag('teams', 'Team profiles')
      .addTag('players', 'Player profiles')
      .addTag('standings', 'League standings tables')
      .build();

    const document = SwaggerModule.createDocument(app, config);
    SwaggerModule.setup('api/docs', app, document);
  }

  const port = process.env.PORT ?? 3001;
  await app.listen(port);

  const logger = new Logger('Bootstrap');
  logger.log(`🚀 Sports Collector running on http://localhost:${port}`);
  logger.log(`📖 API docs: http://localhost:${port}/api/docs`);
}

bootstrap();
    const config = new DocumentBuilder()
      .setTitle('Sports Collector API')
      .setDescription('Internal REST API for sports data — fixtures, standings, players, live scores')
      .setVersion('1.0')
      .addTag('matches', 'Match fixtures and live scores')
      .addTag('leagues', 'Leagues and seasons')
      .addTag('teams', 'Team profiles')
      .addTag('players', 'Player profiles')
      .addTag('standings', 'League standings tables')
      .build();

    const document = SwaggerModule.createDocument(app, config);
    SwaggerModule.setup('api/docs', app, document);
  }

  const port = process.env.PORT ?? 3001;
  await app.listen(port);

  const logger = new Logger('Bootstrap');
  logger.log(`🚀 Sports Collector running on http://localhost:${port}`);
  logger.log(`📖 API docs: http://localhost:${port}/api/docs`);
}

bootstrap();
