import pygame, sys
from PIL import Image
pygame.init()

def displayImage( screen, px, topleft):
    screen.blit(px, px.get_rect())
    if topleft:
        pygame.draw.rect( screen, 0, pygame.Rect(topleft[0], topleft[1], pygame.mouse.get_pos()[0] - topleft[0], pygame.mouse.get_pos()[1] - topleft[1]))
    pygame.display.flip()

def setup(path):
    px = pygame.image.load(path)
    screen = pygame.display.set_mode( px.get_rect()[2:] )
    screen.blit(px, px.get_rect())
    pygame.display.flip()
    return screen, px

def mainLoop(screen, px):
    topleft = None
    bottomright = None
    runProgram = True
    while runProgram:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                runProgram = False
            elif event.type == pygame.MOUSEBUTTONUP:
                if not topleft:
                    topleft = event.pos
                else:
                    bottomright = event.pos
                    runProgram = False
        displayImage(screen, px, topleft)
    return ( topleft + bottomright )


if __name__ == "__main__":
    screen, px = setup(sys.argv[1])
    left, upper, right, lower = mainLoop(screen, px)
    im = Image.open(sys.argv[1])
    im = im.crop(( left, upper, right, lower))
    im.save(sys.argv[2])