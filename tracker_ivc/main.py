import cv2
from ultralytics import YOLO
import logging
import pygame, sys, random
from tracker import open_camera, object_tracking

pygame.init()

WIDTH, HEIGHT = 1280, 720

FONT = pygame.font.SysFont("Consolas", int(WIDTH / 20))

SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong!")

CLOCK = pygame.time.Clock()

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

player = pygame.Rect(0, 0, 10, 100)
player.center = (WIDTH - 100, HEIGHT / 2)

opponent = pygame.Rect(0, 0, 10, 100)
opponent.center = (100, HEIGHT / 2)

ball = pygame.Rect(0, 0, 20, 20)
ball.center = (WIDTH / 2, HEIGHT / 2)

player_score, opponent_score = 0, 0
x_speed, y_speed = 1, 1

MENU = "menu"
GAME = "game"
END = "end"
current_state = MENU

capture = open_camera()

def display_menu():
    SCREEN.fill(BLACK)
    title_text = FONT.render("PONG GAME", True, WHITE)
    start_text = FONT.render("Press SPACE to Start", True, WHITE)
    SCREEN.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, HEIGHT // 3))
    SCREEN.blit(start_text, (WIDTH // 2 - start_text.get_width() // 2, HEIGHT // 2))
    pygame.display.update()

def display_end(winner):
    SCREEN.fill(BLACK)
    winner_text = FONT.render(f"{winner} WINS!", True, WHITE)
    restart_text = FONT.render("Press SPACE to Restart", True, WHITE)
    exit_text = FONT.render("Press ESC to Exit", True, WHITE)
    SCREEN.blit(winner_text, (WIDTH // 2 - winner_text.get_width() // 2, HEIGHT // 3))
    SCREEN.blit(restart_text, (WIDTH // 2 - restart_text.get_width() // 2, HEIGHT // 2))
    SCREEN.blit(exit_text, (WIDTH // 2 - exit_text.get_width() // 2, HEIGHT // 2 + 50))
    pygame.display.update()

def reset_game():
    global player_score, opponent_score, ball, x_speed, y_speed
    player_score, opponent_score = 0, 0
    ball.center = (WIDTH / 2, HEIGHT / 2)
    x_speed, y_speed = random.choice([1, -1]), random.choice([1, -1])

# Game loop
while True:
    if current_state == MENU:
        display_menu()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                reset_game()
                current_state = GAME

    elif current_state == GAME:
        frame, center_left, center_right, combined_frame_annotated = object_tracking(capture, HEIGHT)

        # Update player one paddle position
        if center_right and center_right[1] is not None:
            player.centery = center_right[1]

        # confines paddle to screen bounds
        player.top = max(0, min(player.top, HEIGHT - player.height))

        # Update player 2 (opponent) paddle position
        if center_left and center_left[1] is not None:
            opponent.centery = center_left[1]

        # confines paddle to screen bounds
        opponent.top = max(0, min(opponent.top, HEIGHT - opponent.height))

        # Ball movement and collision
        if ball.y >= HEIGHT:
            y_speed = -1
        if ball.y <= 0:
            y_speed = 1
        if ball.x <= 0:
            player_score += 1
            ball.center = (WIDTH / 2, HEIGHT / 2)
            x_speed, y_speed = random.choice([1, -1]), random.choice([1, -1])
        if ball.x >= WIDTH:
            opponent_score += 1
            ball.center = (WIDTH / 2, HEIGHT / 2)
            x_speed, y_speed = random.choice([1, -1]), random.choice([1, -1])
        if player.x - ball.width <= ball.x <= player.right and player.top <= ball.y <= player.bottom:
            x_speed = -1
        if opponent.x - ball.width <= ball.x <= opponent.right and opponent.top <= ball.y <= opponent.bottom:
            x_speed = 1

        ball.x += x_speed * 25
        ball.y += y_speed * 25

        # Render to screen game objects
        SCREEN.fill(BLACK)
        pygame.draw.rect(SCREEN, WHITE, player)
        pygame.draw.rect(SCREEN, WHITE, opponent)
        pygame.draw.circle(SCREEN, WHITE, ball.center, 10)

        player_score_text = FONT.render(str(player_score), True, WHITE)
        opponent_score_text = FONT.render(str(opponent_score), True, WHITE)
        SCREEN.blit(player_score_text, (WIDTH // 2 + 50, 50))
        SCREEN.blit(opponent_score_text, (WIDTH // 2 - 50, 50))

        # Display camera tracking
        if frame is not None:
            cv2.imshow("Object Tracking", combined_frame_annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        pygame.display.update()
        CLOCK.tick(60)

        # Check for end condition
        if player_score >= 5:
            current_state = END
            winner = "Player"
        elif opponent_score >= 5:
            current_state = END
            winner = "Opponent"

    elif current_state == END:
        display_end(winner)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    current_state = MENU
                    reset_game()
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

capture.release()
cv2.destroyAllWindows()
pygame.quit()
sys.exit()