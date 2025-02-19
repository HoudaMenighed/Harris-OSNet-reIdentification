import pygame
import sys
import cv2
import pygame.freetype
import os
import webbrowser

pygame.init()

script_dir10000 = os.path.dirname(os.path.abspath(__file__))
image_path10000 = os.path.join(script_dir10000, "images\\track.jpg")
icon = pygame.image.load(image_path10000)
pygame.display.set_icon(icon)


def get_font(size):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    font_path = os.path.join(script_dir, "Night Monday.otf")

    try:
        return pygame.font.Font(font_path, size)
    except FileNotFoundError:
        print(f"Cannot find font file: {font_path}")
        raise


SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 660
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "images\\15.jpeg")

BG = pygame.image.load(image_path)
BG = pygame.transform.scale(BG, (SCREEN_WIDTH, SCREEN_HEIGHT))

script_dir1 = os.path.dirname(os.path.abspath(__file__))
image_path1 = os.path.join(script_dir1, "images\\play.png")
image = pygame.image.load(image_path1)
resized_image = pygame.transform.scale(image, (75, 80))
image_position = (106, 265)

script_dir2 = os.path.dirname(os.path.abspath(__file__))
image_path2 = os.path.join(script_dir2, "images\\exit.png")
image2 = pygame.image.load(image_path2)
resized_image2 = pygame.transform.scale(image2, (65, 65))
image_position2 = (112, 460)


def main_menu():
    while True:
        screen.blit(BG, (0, 0))

        start_button_font = get_font(130)
        start_button_color = (93, 187, 245)
        pygame.draw.rect(screen, (11, 20, 143), (95, 245, 300, 120))
        pygame.draw.rect(screen, (235, 19, 48), (217, 251, 170, 3))
        pygame.draw.rect(screen, (235, 19, 48), (387, 251, 3, 67))
        pygame.draw.rect(screen, (247, 230, 153), (88, 373, 170, 3))
        pygame.draw.rect(screen, (247, 230, 153), (85, 309, 3, 67))

        start_button_rect = pygame.Rect(110, 260, 270, 90)
        pygame.draw.rect(screen, start_button_color, start_button_rect)
        start_text = start_button_font.render("Start", True, (0, 0, 0))
        screen.blit(start_text, (start_button_rect.x + 70, 245))

        screen.blit(resized_image, image_position)

        exit_button_font = get_font(130)
        exit_button_color = (93, 187, 245)
        pygame.draw.rect(screen, (11, 20, 143), (95, 430, 300, 120))
        pygame.draw.rect(screen, (235, 19, 48), (217, 435, 170, 3))
        pygame.draw.rect(screen, (235, 19, 48), (387, 435, 3, 67))
        pygame.draw.rect(screen, (247, 230, 153), (88, 559, 170, 3))
        pygame.draw.rect(screen, (247, 230, 153), (85, 495, 3, 67))

        exit_button_rect = pygame.Rect(110, 445, 270, 90)
        pygame.draw.rect(screen, exit_button_color, exit_button_rect)
        exit_text = exit_button_font.render("Exit", True, (0, 0, 0))
        screen.blit(exit_text, (exit_button_rect.x + 95, 430))

        screen.blit(resized_image2, image_position2)

        text_font = get_font(60)
        text = text_font.render(" Deep Learning based Multi-Object ", True, (55, 120, 250))
        text1 = text_font.render("Tracking System", True, (55, 120, 250))

        screen.blit(text, (36, 50))
        screen.blit(text1, (150, 95))

        pygame.draw.rect(screen, (235, 19, 48), (147, 150, 275, 3))
        pygame.draw.rect(screen, (235, 19, 48), (37, 69, 3, 78))
        pygame.draw.rect(screen, (247, 230, 153), (243, 45, 263, 3))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if exit_button_rect.collidepoint(event.pos):
                    pygame.quit()
                    sys.exit()

                if start_button_rect.collidepoint(event.pos):
                    Start()

        pygame.display.update()


script_dir3 = os.path.dirname(os.path.abspath(__file__))
image_path3 = os.path.join(script_dir3, "images\\i6.jpg")
BG2 = pygame.image.load(image_path3)
BG2 = pygame.transform.scale(BG2, (SCREEN_WIDTH, SCREEN_HEIGHT))

script_dir4 = os.path.dirname(os.path.abspath(__file__))
image_path4 = os.path.join(script_dir4, "images\\motion-sensor.png")
image3 = pygame.image.load(image_path4)
resized_image3 = pygame.transform.scale(image3, (44, 44))
image_position3 = (262, 347)

script_dir5 = os.path.dirname(os.path.abspath(__file__))
image_path5 = os.path.join(script_dir5, "images\\exit2.png")
image5 = pygame.image.load(image_path5)
resized_image5 = pygame.transform.scale(image5, (24, 23))
image_position5 = (500, 581)

script_dir6 = os.path.dirname(os.path.abspath(__file__))
image_path6 = os.path.join(script_dir6, "images\\left-arrow.png")
image6 = pygame.image.load(image_path6)
resized_image6 = pygame.transform.scale(image6, (37, 38))
image_position6 = (5, 3)

script_dir7 = os.path.dirname(os.path.abspath(__file__))
image_path7 = os.path.join(script_dir7, "images\\question.png")
img7 = pygame.image.load(image_path7)
resized_img7 = pygame.transform.scale(img7, (24, 27))
img_position7 = (154, 579)

script_dir8 = os.path.dirname(os.path.abspath(__file__))
image_path8 = os.path.join(script_dir8, "images\\courriel-de-contact.png")
img8 = pygame.image.load(image_path8)
resized_img8 = pygame.transform.scale(img8, (29, 30))
img_position8 = (301, 577)

script_dir9 = os.path.dirname(os.path.abspath(__file__))
image_path9 = os.path.join(script_dir9, "images\\training.png")
img9 = pygame.image.load(image_path9)
resized_img9 = pygame.transform.scale(img9, (38, 45))
img_position9 = (265, 193)


def Start():
    screen.blit(BG2, (0, 0))

    text_font = get_font(80)
    text = text_font.render("Multi-Object Tracking", True, (72, 171, 247))
    screen.blit(text, (344, 5))
    pygame.draw.rect(screen, (242, 17, 51), (370, 64, 437, 3))

    pygame.draw.rect(screen, (159, 224, 252), (255, 338, 240, 65))
    pygame.draw.rect(screen, (233, 250, 227), (262, 344, 45, 53))

    tracking_button_font = get_font(60)
    tracking_button_color = (161, 207, 145)
    tracking_button_rect = pygame.Rect(315, 348, 174, 46)
    pygame.draw.rect(screen, tracking_button_color, tracking_button_rect)
    tracking_text = tracking_button_font.render("Tracking", True, (0, 0, 0))
    pygame.draw.rect(screen, (233, 250, 227), (320, 352, 164, 37))

    screen.blit(tracking_text, (334, 344))

    screen.blit(resized_image3, image_position3)

    pygame.draw.rect(screen, (159, 224, 252), (255, 183, 240, 65))
    pygame.draw.rect(screen, (233, 250, 227), (262, 189, 45, 53))

    training_button_font = get_font(60)
    training_button_color = (161, 207, 145)
    training_button_rect = pygame.Rect(315, 193, 174, 46)

    pygame.draw.rect(screen, training_button_color, training_button_rect)
    training_text = training_button_font.render("Training", True, (0, 0, 0))
    pygame.draw.rect(screen, (233, 250, 227), (320, 197, 164, 37))

    screen.blit(training_text, (331, 188))

    screen.blit(resized_img9, img_position9)

    pygame.draw.rect(screen, (159, 224, 252), (492, 572, 112, 43))
    pygame.draw.rect(screen, (233, 250, 227), (497, 576, 30, 34))
    pygame.draw.rect(screen, (161, 207, 145), (534, 576, 66, 34))

    exit_button_font = get_font(44)
    exit_button_color = (233, 250, 227)
    exit_button_rect = pygame.Rect(538, 579, 58, 27)
    pygame.draw.rect(screen, exit_button_color, exit_button_rect)
    exit_text = exit_button_font.render("Exit", True, (0, 0, 0))
    screen.blit(exit_text, (543, 572))

    screen.blit(resized_image5, image_position5)

    back_button_font = get_font(66)
    back_button_color = (233, 250, 227)
    back_button_rect = pygame.Rect(7, 7, 36, 30)
    pygame.draw.rect(screen, back_button_color, back_button_rect)
    back_text = back_button_font.render("", True, (0, 0, 0))
    screen.blit(back_text, (back_button_rect.x + 3, 564))

    screen.blit(resized_image6, image_position6)

    pygame.draw.rect(screen, (159, 224, 252), (147, 572, 112, 43))
    pygame.draw.rect(screen, (233, 250, 227), (152, 576, 30, 34))
    pygame.draw.rect(screen, (161, 207, 145), (189, 576, 66, 34))

    help_button_font = get_font(44)
    help_button_color = (233, 250, 227)
    help_button_rect = pygame.Rect(193, 579, 58, 27)
    pygame.draw.rect(screen, help_button_color, help_button_rect)
    help_text = help_button_font.render("Help", True, (0, 0, 0))
    screen.blit(help_text, (198, 572))

    screen.blit(resized_img7, img_position7)

    pygame.draw.rect(screen, (159, 224, 252), (295, 572, 161, 43))
    pygame.draw.rect(screen, (233, 250, 227), (302, 576, 30, 34))
    pygame.draw.rect(screen, (161, 207, 145), (339, 576, 112, 34))

    contact_button_font = get_font(44)
    contact_button_color = (233, 250, 227)
    contact_button_rect = pygame.Rect(344, 579, 102, 27)
    pygame.draw.rect(screen, contact_button_color, contact_button_rect)
    contact_text = contact_button_font.render("Contact", True, (0, 0, 0))
    screen.blit(contact_text, (contact_button_rect.x + 3, 572))

    screen.blit(resized_img8, img_position8)

    pygame.draw.rect(screen, (242, 17, 51), (393, 326, 110, 4))
    pygame.draw.rect(screen, (242, 17, 51), (501, 326, 4, 42))
    pygame.draw.rect(screen, (126, 227, 120), (245, 409, 110, 4))
    pygame.draw.rect(screen, (126, 227, 120), (245, 370, 4, 42))

    pygame.draw.rect(screen, (242, 17, 51), (393, 171, 110, 4))
    pygame.draw.rect(screen, (242, 17, 51), (501, 171, 4, 42))
    pygame.draw.rect(screen, (126, 227, 120), (245, 255, 110, 4))
    pygame.draw.rect(screen, (126, 227, 120), (245, 217, 4, 42))

    pygame.display.update()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if exit_button_rect.collidepoint(event.pos):
                    pygame.quit()
                    sys.exit()

                elif back_button_rect.collidepoint(event.pos):
                    main_menu()

                elif tracking_button_rect.collidepoint(event.pos):
                    Tracking()
                elif contact_button_rect.collidepoint(event.pos):
                    Contact()
                elif training_button_rect.collidepoint(event.pos):
                    Training()
                elif help_button_rect.collidepoint(event.pos):
                    Help()


script_dir10 = os.path.dirname(os.path.abspath(__file__))
image_path10 = os.path.join(script_dir10, "images\\left-arrow.png")
image7 = pygame.image.load(image_path10)
resized_image7 = pygame.transform.scale(image6, (44, 45))
image_position7 = (1040, 597)

script_dir11 = os.path.dirname(os.path.abspath(__file__))
image_path11 = os.path.join(script_dir11, "images\\end.png")
image10 = pygame.image.load(image_path11)
resized_image10 = pygame.transform.scale(image10, (40, 40))
image_position10 = (659, 602)

script_dir12 = os.path.dirname(os.path.abspath(__file__))
image_path12 = os.path.join(script_dir12, "images\\n7.jpeg")
BG3 = pygame.image.load(image_path12)
BG3 = pygame.transform.scale(BG3, (SCREEN_WIDTH, SCREEN_HEIGHT))

script_dir13 = os.path.dirname(os.path.abspath(__file__))
image_path13 = os.path.join(script_dir13, "images\\video.png")
image14 = pygame.image.load(image_path13)
resized_image14 = pygame.transform.scale(image14, (90, 79))
image_position14 = (781, 236)

script_dir14 = os.path.dirname(os.path.abspath(__file__))
image_path14 = os.path.join(script_dir14, "images\\camera.png")
image15 = pygame.image.load(image_path14)
resized_image15 = pygame.transform.scale(image15, (90, 80))
image_position15 = (275, 425)

script_dir15 = os.path.dirname(os.path.abspath(__file__))
image_path15 = os.path.join(script_dir15, "images\\exit2.png")
image16 = pygame.image.load(image_path15)
resized_image16 = pygame.transform.scale(image16, (51, 46))
image_position16 = (1135, 592)


def Tracking():
    screen.blit(BG3, (0, 0))

    text_font = get_font2(50)
    text = text_font.render("Multi-Object Tracking", True, (154, 215, 245))
    screen.blit(text, (250, 35))

    pygame.draw.rect(screen, (232, 250, 132), (239, 100, 85, 4))
    pygame.draw.rect(screen, (232, 250, 132), (239, 59, 4, 42))

    pygame.draw.rect(screen, (174, 227, 163), (230, 110, 50, 4))
    pygame.draw.rect(screen, (174, 227, 163), (230, 84, 4, 30))

    pygame.draw.rect(screen, (232, 250, 132), (841, 40, 85, 4))
    pygame.draw.rect(screen, (232, 250, 132), (925, 40, 4, 42))

    pygame.draw.rect(screen, (174, 227, 163), (888, 30, 50, 4))
    pygame.draw.rect(screen, (174, 227, 163), (934, 30, 4, 30))

    pygame.draw.rect(screen, (252, 251, 235), (260, 231, 240, 65))
    pygame.draw.rect(screen, (252, 251, 235), (480, 257, 240, 65))

    pygame.draw.rect(screen, (252, 251, 235), (776, 240, 100, 71))

    screen.blit(resized_image14, image_position14)

    track1_button_font = get_font2(39)
    track1_button_color = (139, 220, 252)
    track1_button_rect = pygame.Rect(270, 240, 440, 73)
    pygame.draw.rect(screen, track1_button_color, track1_button_rect)
    track1_text = track1_button_font.render("Track from Video ", True, (0, 0, 0))
    screen.blit(track1_text, (283, 247))

    pygame.draw.rect(screen, (252, 251, 235), (425, 421, 240, 65))
    pygame.draw.rect(screen, (252, 251, 235), (697, 447, 240, 65))

    pygame.draw.rect(screen, (252, 251, 235), (270, 430, 100, 71))

    track2_button_font = get_font2(39)
    track2_button_color = (139, 220, 252)
    track2_button_rect = pygame.Rect(435, 430, 492, 73)
    pygame.draw.rect(screen, track2_button_color, track2_button_rect)
    track2_text = track2_button_font.render("Track from Camera ", True, (0, 0, 0))
    screen.blit(track2_text, (447, 437))

    screen.blit(resized_image15, image_position15)

    back_button_font = get_font(66)
    back_button_color = (233, 250, 227)
    back_button_rect = pygame.Rect(7, 7, 36, 30)
    pygame.draw.rect(screen, back_button_color, back_button_rect)
    back_text = back_button_font.render("", True, (0, 0, 0))
    screen.blit(back_text, (back_button_rect.x + 3, 564))

    screen.blit(resized_image6, image_position6)

    exit_button_font = get_font(44)
    exit_button_color = (233, 250, 227)
    exit_button_rect = pygame.Rect(1132, 590, 57, 51)
    pygame.draw.rect(screen, exit_button_color, exit_button_rect)
    exit_text = exit_button_font.render("", True, (0, 0, 0))
    screen.blit(exit_text, (543, 572))

    screen.blit(resized_image16, image_position16)

    pygame.display.update()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:

                if back_button_rect.collidepoint(event.pos):
                    Start()
                if exit_button_rect.collidepoint(event.pos):
                    pygame.quit()
                    sys.exit()
                if track1_button_rect.collidepoint(event.pos):
                    TrackOption()
                if track2_button_rect.collidepoint(event.pos):
                    TrackCamera()


def TrackOption():
    screen.blit(BG3, (0, 0))
    text_font = get_font2(50)
    text = text_font.render("Tracking Options", True, (252, 20, 20))
    screen.blit(text, (320, 10))

    pygame.draw.rect(screen, (70, 250, 25), (289, 75, 85, 4))
    pygame.draw.rect(screen, (70, 250, 25), (289, 34, 4, 42))

    pygame.draw.rect(screen, (250, 250, 37), (280, 85, 50, 4))
    pygame.draw.rect(screen, (250, 250, 37), (280, 59, 4, 30))

    pygame.draw.rect(screen, (70, 250, 25), (801, 15, 85, 4))
    pygame.draw.rect(screen, (70, 250, 25), (885, 15, 4, 42))

    pygame.draw.rect(screen, (250, 250, 37), (848, 5, 50, 4))
    pygame.draw.rect(screen, (250, 250, 37), (894, 5, 4, 30))

    text_font = get_font3(47)
    text = text_font.render("Choose :", True, (250, 250, 37))
    screen.blit(text, (20, 125))
    pygame.draw.rect(screen, (70, 250, 25), (36, 190, 150, 4))

    text_font = get_font3(43)
    text = text_font.render("Tracking :", True, (9, 214, 33))
    screen.blit(text, (115, 265))
    pygame.draw.rect(screen, (218, 237, 76), (143, 328, 126, 3))

    center_y = SCREEN_HEIGHT // 2 + 100

    start_pos = (0, center_y)
    end_pos = (SCREEN_WIDTH, center_y)
    black = (0, 0, 0)
    white = (255, 255, 255)

    pygame.draw.line(screen, black, start_pos, end_pos, 5)

    text_font = get_font3(43)
    text = text_font.render("Evaluation :", True, (9, 214, 33))
    screen.blit(text, (115, 485))
    pygame.draw.rect(screen, (218, 237, 76), (145, 548, 145, 3))



    back_button_font = get_font(66)
    back_button_color = (233, 250, 227)
    back_button_rect = pygame.Rect(7, 7, 36, 30)
    pygame.draw.rect(screen, back_button_color, back_button_rect)
    back_text = back_button_font.render("", True, (0, 0, 0))
    screen.blit(back_text, (back_button_rect.x + 3, 564))

    screen.blit(resized_image6, image_position6)

    exit_button_font = get_font(44)
    exit_button_color = (233, 250, 227)
    exit_button_rect = pygame.Rect(1132, 590, 57, 51)
    pygame.draw.rect(screen, exit_button_color, exit_button_rect)
    exit_text = exit_button_font.render("", True, (0, 0, 0))
    screen.blit(exit_text, (543, 572))

    screen.blit(resized_image16, image_position16)

    option1_button_font = get_font2(27)
    option1_button_color = (253, 255, 158)
    option1_button_rect = pygame.Rect(480, 220, 465, 70)

    pygame.draw.rect(screen, option1_button_color, option1_button_rect)
    option1_text = option1_button_font.render("Start the Video Tracking", True, (0, 0, 0))
    pygame.draw.rect(screen, (186, 232, 224), (490, 230, 445, 50))
    screen.blit(option1_text, (506, 237))

    option2_button_font = get_font2(27)
    option2_button_color = (253, 255, 158)
    option2_button_rect = pygame.Rect(480, 445, 465, 70)

    pygame.draw.rect(screen, option2_button_color, option2_button_rect)
    option2_text = option2_button_font.render("Evaluate Our Model ", True, (0, 0, 0))
    pygame.draw.rect(screen, (186, 232, 224), (490, 455, 445, 50))
    screen.blit(option2_text, (545, 462))

    option3_button_font = get_font2(27)
    option3_button_color = (253, 255, 158)
    option3_button_rect = pygame.Rect(440, 310, 550, 70)

    pygame.draw.rect(screen, option3_button_color, option3_button_rect)
    option3_text = option3_button_font.render("Show already processed video", True, (0, 0, 0))
    pygame.draw.rect(screen, (186, 232, 224), (450, 320, 530, 50))
    screen.blit(option3_text, (460, 327))

    option4_button_font = get_font2(27)
    option4_button_color = (253, 255, 158)
    option4_button_rect = pygame.Rect(440, 535, 550, 70)

    pygame.draw.rect(screen, option4_button_color, option4_button_rect)
    option4_text = option4_button_font.render("Show already evaluated video", True, (0, 0, 0))
    pygame.draw.rect(screen, (186, 232, 224), (450, 545, 530, 50))
    screen.blit(option4_text, (463, 552))

    pygame.display.update()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:

                if back_button_rect.collidepoint(event.pos):
                    Tracking()

                elif option1_button_rect.collidepoint(event.pos):
                    TrackVideo()

                elif option2_button_rect.collidepoint(event.pos):
                    DoEvaluation()

                elif option3_button_rect.collidepoint(event.pos):
                    ShowVideo()

                elif option4_button_rect.collidepoint(event.pos):
                    ShowEvaluation()



                elif exit_button_rect.collidepoint(event.pos):
                    pygame.quit()
                    sys.exit()


script_dir160 = os.path.dirname(os.path.abspath(__file__))
image_path160 = os.path.join(script_dir160, "images\\22.jpg")
BG3000 = pygame.image.load(image_path160)
BG3000 = pygame.transform.scale(BG3000, (SCREEN_WIDTH, SCREEN_HEIGHT))

script_dir150 = os.path.dirname(os.path.abspath(__file__))
image_path150 = os.path.join(script_dir150, "images\\cursor.png")
image160 = pygame.image.load(image_path150)
resized_image160 = pygame.transform.scale(image160, (45, 43))
image_position160 = (72, 589)

script_dir151 = os.path.dirname(os.path.abspath(__file__))
image_path151 = os.path.join(script_dir151, "images\\video-pause-button.png")
image161 = pygame.image.load(image_path151)
resized_image161 = pygame.transform.scale(image161, (39, 35))
image_position161 = (454, 593)

script_dir152 = os.path.dirname(os.path.abspath(__file__))
image_path152 = os.path.join(script_dir152, "images\\exit2.png")
image162 = pygame.image.load(image_path152)
resized_image162 = pygame.transform.scale(image162, (32, 25))
image_position162 = (1155, 9)

script_dir153 = os.path.dirname(os.path.abspath(__file__))
image_path153 = os.path.join(script_dir153, "images\\plus.png")
image163 = pygame.image.load(image_path153)
resized_image163 = pygame.transform.scale(image163, (30, 24))
image_position163 = (874, 598)

script_dir154 = os.path.dirname(os.path.abspath(__file__))
image_path154 = os.path.join(script_dir154, "images\\minimize-sign.png")
image164 = pygame.image.load(image_path154)
resized_image164 = pygame.transform.scale(image164, (30, 28))
image_position164 = (1075, 596)




def ShowVideo():
    screen.blit(BG3000, (0, 0))

    text_font = get_font2(50)
    text = text_font.render("Video Already Processed", True, (155, 242, 131))
    screen.blit(text, (210, 6))

    back_button_font = get_font(66)
    back_button_color = (233, 250, 227)
    back_button_rect = pygame.Rect(7, 7, 36, 30)
    pygame.draw.rect(screen, back_button_color, back_button_rect)
    back_text = back_button_font.render("", True, (0, 0, 0))
    screen.blit(back_text, (back_button_rect.x + 3, 564))

    screen.blit(resized_image6, image_position6)

    exit_button_font = get_font(44)
    exit_button_color = (233, 250, 227)
    exit_button_rect = pygame.Rect(1153, 7, 36, 30)
    pygame.draw.rect(screen, exit_button_color, exit_button_rect)
    exit_text = exit_button_font.render("", True, (0, 0, 0))
    screen.blit(exit_text, (543, 572))

    screen.blit(resized_image162, image_position162)

    select_button_font = get_font(65)
    select_button_color = (119, 199, 247)
    select_button_rect = pygame.Rect(70, 585, 280, 50)
    pygame.draw.rect(screen, select_button_color, select_button_rect)
    select_text = select_button_font.render("Select video ", True, (0, 0, 0))
    screen.blit(select_text, (select_button_rect.x + 57, 580))

    screen.blit(resized_image160, image_position160)

    pause_button_font = get_font(65)
    pause_button_color = (119, 199, 247)
    pause_button_rect = pygame.Rect(450, 585, 298, 50)
    pygame.draw.rect(screen, pause_button_color, pause_button_rect)
    pause_text = pause_button_font.render("Pause/Resume", True, (0, 0, 0))
    screen.blit(pause_text, (pause_button_rect.x + 57, 580))

    screen.blit(resized_image161, image_position161)

    pygame.draw.rect(screen, (119, 199, 247), (851, 585, 280, 50))
    text_font = get_font(65)
    text = text_font.render("Speed", True, (0, 0, 0))
    screen.blit(text, (944, 580))

    plus_button_font = get_font(66)
    plus_button_color = (250, 247, 187)
    plus_button_rect = pygame.Rect(870, 594, 41, 33)
    pygame.draw.rect(screen, plus_button_color, plus_button_rect)
    plus_text = plus_button_font.render("", True, (0, 0, 0))
    screen.blit(plus_text, (plus_button_rect.x + 3, 564))

    screen.blit(resized_image163, image_position163)

    less_button_font = get_font(66)
    less_button_color = (250, 247, 187)
    less_button_rect = pygame.Rect(1070, 594, 41, 33)
    pygame.draw.rect(screen, less_button_color, less_button_rect)
    less_text = less_button_font.render("", True, (0, 0, 0))
    screen.blit(less_text, (less_button_rect.x + 3, 564))

    screen.blit(resized_image164, image_position164)

    pygame.display.update()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if back_button_rect.collidepoint(event.pos):
                    TrackOption()
                elif exit_button_rect.collidepoint(event.pos):
                    pygame.quit()
                    sys.exit()
                elif select_button_rect.collidepoint(event.pos):
                    root = Tk()
                    root.withdraw()
                    file_path = filedialog.askopenfilename(title="Select a video file",
                                                           filetypes=[("Video Files", "*.mp4 *.avi *.mkv")])
                    root.destroy()
                    print("Selected file:", file_path)

                    display_video(file_path)
        pygame.display.update()


playback_speed = 0.5

def display_video(file_path):
    cap = cv2.VideoCapture(file_path)
    paused = False


    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    paused = False
    speed_factor = 1.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Original video dimensions: {width} x {height}")
    frame_delay = int(1000 / (cap.get(cv2.CAP_PROP_FPS) * playback_speed))

    back_button_rect = pygame.Rect(7, 7, 36, 30)
    select_button_rect = pygame.Rect(70, 585, 280, 50)
    exit_button_rect = pygame.Rect(1153, 7, 36, 30)
    less_button_rect = pygame.Rect(1070, 594, 41, 33)
    plus_button_rect = pygame.Rect(870, 594, 41, 33)


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (1170, 423))

        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_pygame = pygame.image.frombuffer(frame_rgb.tostring(), (1170, 423), "RGB")

        if frame_pygame:
            screen.blit(frame_pygame, (15,117))

        pygame.display.update()
        pygame.time.wait(frame_delay)


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if back_button_rect.collidepoint(event.pos):
                    TrackOption()
                elif exit_button_rect.collidepoint(event.pos):
                    pygame.quit()
                    sys.exit()
                elif select_button_rect.collidepoint(event.pos):
                    root = Tk()
                    root.withdraw()
                    file_path = filedialog.askopenfilename(title="Select a video file",
                                                           filetypes=[("Video Files", "*.mp4 *.avi *.mkv")])
                    root.destroy()
                    print("Selected file:", file_path)

                    display_video(file_path)
                elif image_position161[0] <= event.pos[0] <= image_position161[0] + resized_image161.get_width() and \
                   image_position161[1] <= event.pos[1] <= image_position161[1] + resized_image161.get_height():
                    paused = not paused
                elif less_button_rect.collidepoint(event.pos):
                    speed_factor -= 0.1
                elif plus_button_rect.collidepoint(event.pos):
                    speed_factor += 0.1

        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + speed_factor)

        if not paused:
            continue

        while paused:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if back_button_rect.collidepoint(event.pos):
                        TrackOption()
                    elif exit_button_rect.collidepoint(event.pos):
                        pygame.quit()
                        sys.exit()
                    elif select_button_rect.collidepoint(event.pos):
                        root = Tk()
                        root.withdraw()
                        file_path = filedialog.askopenfilename(title="Select a video file",
                                                               filetypes=[("Video Files", "*.mp4 *.avi *.mkv")])
                        root.destroy()
                        print("Selected file:", file_path)

                        display_video(file_path)


                    elif image_position161[0] <= event.pos[0] <= image_position161[0] + resized_image161.get_width() and \
                       image_position161[1] <= event.pos[1] <= image_position161[1] + resized_image161.get_height():
                        paused = False
                    elif less_button_rect.collidepoint(event.pos):
                        speed_factor -= 0.1
                    elif plus_button_rect.collidepoint(event.pos):
                        speed_factor += 0.1



    cap.release()

from moviepy.editor import VideoFileClip, vfx
from tkinter import Tk
from tkinter.filedialog import askopenfilename

script_dir1771 = os.path.dirname(os.path.abspath(__file__))
image_path1771 = os.path.join(script_dir1771, "images\\h24.jpg")
BG3001 = pygame.image.load(image_path1771)
BG3001 = pygame.transform.scale(BG3001, (SCREEN_WIDTH, SCREEN_HEIGHT))

script_dir155 = os.path.dirname(os.path.abspath(__file__))
image_path155 = os.path.join(script_dir155, "images\\show-video.png")
image169 = pygame.image.load(image_path155)
resized_image169 = pygame.transform.scale(image169, (58, 46))
image_position169 = (46, 280)

script_dir156 = os.path.dirname(os.path.abspath(__file__))
image_path156 = os.path.join(script_dir156, "images\\show-video.png")
image1696 = pygame.image.load(image_path156)
resized_image1696 = pygame.transform.scale(image1696, (58, 46))
image_position1696 = (46, 368)

import numpy as np
from tkinter import ttk
import cv2
from PIL import Image, ImageTk



def video_mot1704(video_path_mot1704):
    cap = cv2.VideoCapture(video_path_mot1704)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    window_width, window_height = 950, 590
    video_width, video_height = 950, 590

    root = tk.Tk()
    root.title("Video")
    root.geometry(f"{window_width}x{window_height}")

    video_label = tk.Label(root)
    video_label.pack()

    global exit_flag
    exit_flag = False

    def exit_callback():
        global exit_flag
        exit_flag = True
        root.quit()

    exit_button = ttk.Button(root, text="Exit", command=exit_callback)
    exit_button.pack()

    frame_count = 0
    speed_factor = 2  # To slow down by 0.5x

    def show_frame():
        global exit_flag
        nonlocal frame_count
        if exit_flag:
            return

        ret, frame = cap.read()
        if not ret:
            return

        if frame_count % speed_factor == 0:  # Display every nth frame
            resized_frame = cv2.resize(frame, (video_width, video_height))

            frame_image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            frame_image = Image.fromarray(frame_image)
            frame_image = ImageTk.PhotoImage(frame_image)

            video_label.config(image=frame_image)
            video_label.image = frame_image

        frame_count += 1
        root.after(25, show_frame)

    show_frame()

    root.mainloop()

    cap.release()
    cv2.destroyAllWindows()



def video_mot1709(video_path_mot1709):
    cap = cv2.VideoCapture(video_path_mot1709)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    window_width, window_height = 950, 590
    video_width, video_height = 950, 590

    root = tk.Tk()
    root.title("Video")
    root.geometry(f"{window_width}x{window_height}")

    video_label = tk.Label(root)
    video_label.pack()

    global exit_flag
    exit_flag = False

    def exit_callback():
        global exit_flag
        exit_flag = True
        root.quit()

    exit_button = ttk.Button(root, text="Exit", command=exit_callback)
    exit_button.pack()

    frame_count = 0
    speed_factor = 2  # To slow down by 0.5x

    def show_frame():
        global exit_flag
        nonlocal frame_count
        if exit_flag:
            return

        ret, frame = cap.read()
        if not ret:
            return

        if frame_count % speed_factor == 0:  # Display every nth frame
            resized_frame = cv2.resize(frame, (video_width, video_height))

            frame_image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            frame_image = Image.fromarray(frame_image)
            frame_image = ImageTk.PhotoImage(frame_image)

            video_label.config(image=frame_image)
            video_label.image = frame_image

        frame_count += 1
        root.after(25, show_frame)

    show_frame()

    root.mainloop()

    cap.release()
    cv2.destroyAllWindows()

def load_selected_options(output_file):
    try:
        with open(output_file, "r") as file:
            FP = file.readline().strip()
            FN = file.readline().strip()
            IDS = file.readline().strip()
            GT = file.readline().strip()
            MOTA = file.readline().strip()
            MOTP = file.readline().strip()
            return FP, FN, IDS, GT, MOTA, MOTP
    except FileNotFoundError:
        return None, None, None, None, None, None


def is_file_not_empty(filename):
    return os.path.isfile(filename) and os.path.getsize(filename) > 0


#output_file04 = "C:/Users/STS/PycharmProjects/INTERFACE/output04_metrics.txt"
#output_file09 = "C:/Users/STS/PycharmProjects/INTERFACE/output09_metrics.txt"

filename04 = "output04_metrics.txt"
filename09 = "output09_metrics.txt"

output_file04 = os.path.join(os.getcwd(), filename04)
output_file09 = os.path.join(os.getcwd(), filename09)


def ShowEvaluation():
    screen.blit(BG3001, (0, 0))

    text_font = get_font2(39)
    text = text_font.render("Video Already Evaluated", True, (155, 242, 131))
    screen.blit(text, (310, 8))
    pygame.draw.rect(screen, (255, 43, 54), (341, 60, 520, 4))

    back_button_font = get_font(66)
    back_button_color = (233, 250, 227)
    back_button_rect = pygame.Rect(7, 7, 36, 30)
    pygame.draw.rect(screen, back_button_color, back_button_rect)
    back_text = back_button_font.render("", True, (0, 0, 0))
    screen.blit(back_text, (back_button_rect.x + 3, 564))

    screen.blit(resized_image6, image_position6)

    exit_button_font = get_font(44)
    exit_button_color = (233, 250, 227)
    exit_button_rect = pygame.Rect(1153, 7, 36, 30)
    pygame.draw.rect(screen, exit_button_color, exit_button_rect)
    exit_text = exit_button_font.render("", True, (0, 0, 0))
    screen.blit(exit_text, (543, 572))

    screen.blit(resized_image162, image_position162)

    pygame.draw.rect(screen, (227, 249, 250), (205, 258, 940, 175))
    pygame.draw.rect(screen, (251, 255, 176), (140, 200, 168, 233))
    pygame.draw.rect(screen, (251, 255, 176), (205, 200, 940, 62))

    pygame.draw.rect(screen, (255, 255, 255), (140, 200, 1005, 4))

    text_font = get_font2(30)
    text = text_font.render("Videos", True, (214, 6, 31))
    screen.blit(text, (162, 208))
    pygame.draw.rect(screen, (255, 255, 255), (140, 258, 1005, 4))

    text_font = get_font2(30)
    text = text_font.render("FP", True, (214, 6, 31))
    screen.blit(text, (361, 208))

    text_font = get_font2(30)
    text = text_font.render("FN", True, (214, 6, 31))
    screen.blit(text, (502, 208))

    text_font = get_font2(30)
    text = text_font.render("IDS", True, (214, 6, 31))
    screen.blit(text, (632, 208))

    text_font = get_font2(30)
    text = text_font.render("GT", True, (214, 6, 31))
    screen.blit(text, (763, 208))

    text_font = get_font2(30)
    text = text_font.render("MOTA", True, (214, 6, 31))
    screen.blit(text, (880, 208))

    text_font = get_font2(30)
    text = text_font.render("MOTP", True, (214, 6, 31))
    screen.blit(text, (1028, 208))

    text_font = get_font2(28)
    text = text_font.render("MOT17-04", True, (26, 90, 163))
    screen.blit(text, (147, 280))
    pygame.draw.rect(screen, (255, 255, 255), (140, 342, 1005, 4))

    text_font = get_font2(28)
    text = text_font.render("MOT17-09", True, (26, 90, 163))
    screen.blit(text, (147, 368))
    pygame.draw.rect(screen, (255, 255, 255), (140, 430, 1005, 4))

    pygame.draw.rect(screen, (255, 255, 255), (140, 200, 4, 234))
    pygame.draw.rect(screen, (255, 255, 255), (305, 200, 4, 234))
    pygame.draw.rect(screen, (255, 255, 255), (450, 200, 4, 234))
    pygame.draw.rect(screen, (255, 255, 255), (595, 200, 4, 234))
    pygame.draw.rect(screen, (255, 255, 255), (720, 200, 4, 234))
    pygame.draw.rect(screen, (255, 255, 255), (845, 200, 4, 234))
    pygame.draw.rect(screen, (255, 255, 255), (1000, 200, 4, 234))
    pygame.draw.rect(screen, (255, 255, 255), (1145, 200, 4, 234))

    display1_button_font = get_font(65)
    display1_button_color = (119, 199, 247)
    display1_button_rect = pygame.Rect(38, 278, 75, 50)
    pygame.draw.rect(screen, display1_button_color, display1_button_rect)
    display1_text = display1_button_font.render("", True, (0, 0, 0))
    screen.blit(display1_text, (display1_button_rect.x + 57, 580))

    screen.blit(resized_image169, image_position169)

    display2_button_font = get_font(65)
    display2_button_color = (119, 199, 247)
    display2_button_rect = pygame.Rect(38, 366, 75, 50)
    pygame.draw.rect(screen, display2_button_color, display2_button_rect)
    display2_text = display2_button_font.render("", True, (0, 0, 0))
    screen.blit(display2_text, (display2_button_rect.x + 57, 580))

    screen.blit(resized_image1696, image_position1696)

    pygame.display.update()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if back_button_rect.collidepoint(event.pos):
                    TrackOption()
                elif exit_button_rect.collidepoint(event.pos):
                    pygame.quit()
                    sys.exit()

                elif display1_button_rect.collidepoint(event.pos):

                    video_filename = "MOT1704.mp4"
                    video_folder = "MOT17"

                    video_mot1704_path = os.path.join(os.getcwd(), video_folder, video_filename)
                    #video_mot1704("C:/Users/STS/PycharmProjects/INTERFACE/MOT17/MOT1704.mp4")
                    video_mot1704(video_mot1704_path)
                    FP, FN, IDS, GT, MOTA, MOTP = load_selected_options(output_file04)
                    text_font = get_font2(27)
                    FP_text = text_font.render(str(FP), True, (17, 158, 48))
                    screen.blit(FP_text, (330, 285))

                    FN_text = text_font.render(str(FN), True, (17, 158, 48))
                    screen.blit(FN_text, (472, 285))

                    IDS_text = text_font.render(str(IDS), True, (17, 158, 48))
                    screen.blit(IDS_text, (616, 285))

                    GT_text = text_font.render(str(GT), True, (17, 158, 48))
                    screen.blit(GT_text, (766, 285))

                    MOTA_text = text_font.render(str(MOTA), True, (17, 158, 48))
                    screen.blit(MOTA_text, (862, 285))

                    MOTP_text = text_font.render(str(MOTP), True, (17, 158, 48))
                    screen.blit(MOTP_text, (1011, 285))

                    pygame.display.update()
                elif display2_button_rect.collidepoint(event.pos):
                    video_filename = "MOT1709.mp4"
                    video_folder = "MOT17"
                    #video_mot1709("C:/Users/STS/PycharmProjects/INTERFACE/MOT17/MOT1709.mp4")
                    video_mot1709_path = os.path.join(os.getcwd(), video_folder, video_filename)
                    video_mot1709(video_mot1709_path)
                    FP, FN, IDS, GT, MOTA, MOTP = load_selected_options(output_file09)
                    text_font = get_font2(27)
                    FP_text = text_font.render(str(FP), True, (17, 158, 48))
                    screen.blit(FP_text, (333, 368))

                    FN_text = text_font.render(str(FN), True, (17, 158, 48))
                    screen.blit(FN_text, (477, 368))

                    IDS_text = text_font.render(str(IDS), True, (17, 158, 48))
                    screen.blit(IDS_text, (628, 368))

                    GT_text = text_font.render(str(GT), True, (17, 158, 48))
                    screen.blit(GT_text, (766, 368))

                    MOTA_text = text_font.render(str(MOTA), True, (17, 158, 48))
                    screen.blit(MOTA_text, (857, 368))

                    MOTP_text = text_font.render(str(MOTP), True, (17, 158, 48))
                    screen.blit(MOTP_text, (1011, 368))

                    pygame.display.update()


script_dir16 = os.path.dirname(os.path.abspath(__file__))
image_path16 = os.path.join(script_dir16, "images\\n2.jpg")
BG4 = pygame.image.load(image_path16)
BG4 = pygame.transform.scale(BG4, (SCREEN_WIDTH, SCREEN_HEIGHT))

script_dir17 = os.path.dirname(os.path.abspath(__file__))
image_path17 = os.path.join(script_dir17, "images\\emplacement.png")
image11 = pygame.image.load(image_path17)
resized_image11 = pygame.transform.scale(image11, (65, 70))
image_position11 = (280, 260)

script_dir18 = os.path.dirname(os.path.abspath(__file__))
image_path18 = os.path.join(script_dir18, "images\\mail.png")
image12 = pygame.image.load(image_path18)
resized_image12 = pygame.transform.scale(image12, (65, 70))
image_position12 = (846, 260)

script_dir19 = os.path.dirname(os.path.abspath(__file__))
image_path19 = os.path.join(script_dir19, "images\\courriel-de-contact.png")
image13 = pygame.image.load(image_path19)
resized_image13 = pygame.transform.scale(image13, (79, 86))
image_position13 = (540, 255)


def get_font2(size):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    font_path = os.path.join(script_dir, "BEQINER.otf")

    try:
        return pygame.font.Font(font_path, size)
    except FileNotFoundError:
        print(f"Cannot find font file: {font_path}")
        raise


def get_font3(size):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    font_path = os.path.join(script_dir, "OpenSans-Light.ttf")

    try:
        return pygame.font.Font(font_path, size)
    except FileNotFoundError:
        print(f"Cannot find font file: {font_path}")
        raise


script_dir100 = os.path.dirname(os.path.abspath(__file__))
image_path100 = os.path.join(script_dir100, "images\\h24.jpg")
BG50 = pygame.image.load(image_path100)
BG50 = pygame.transform.scale(BG50, (SCREEN_WIDTH, SCREEN_HEIGHT))

script_dir1000 = os.path.dirname(os.path.abspath(__file__))
image_path1000 = os.path.join(script_dir1000, "images\\train2.png")
BG500 = pygame.image.load(image_path1000)
BG500 = pygame.transform.scale(BG500, (940, 520))

script_dir200 = os.path.dirname(os.path.abspath(__file__))
image_path200 = os.path.join(script_dir200, "images\\next.png")
image130 = pygame.image.load(image_path200)
resized_image130 = pygame.transform.scale(image130, (30, 35))
image_position130 = (1157, 4)

script_dir201 = os.path.dirname(os.path.abspath(__file__))
image_path201 = os.path.join(script_dir201, "images\\yellow.png")
image131 = pygame.image.load(image_path201)
resized_image131 = pygame.transform.scale(image131, (130, 70))
image_position131 = (385, 170)

script_dir202 = os.path.dirname(os.path.abspath(__file__))
image_path202 = os.path.join(script_dir202, "images\\green.png")
image132 = pygame.image.load(image_path202)
resized_image132 = pygame.transform.scale(image132, (130, 70))
image_position132 = (835, 246)

script_dir203 = os.path.dirname(os.path.abspath(__file__))
image_path203 = os.path.join(script_dir203, "images\\green.png")
image133 = pygame.image.load(image_path203)
resized_image133 = pygame.transform.scale(image133, (140, 70))
image_position133 = (848, 380)

script_dir204 = os.path.dirname(os.path.abspath(__file__))
image_path204 = os.path.join(script_dir204, "images\\yellow.png")
image134 = pygame.image.load(image_path204)
resized_image134 = pygame.transform.scale(image134, (130, 70))
image_position134 = (730, 320)

script_dir205 = os.path.dirname(os.path.abspath(__file__))
image_path205 = os.path.join(script_dir205, "images\\yellow.png")
image135 = pygame.image.load(image_path205)
resized_image135 = pygame.transform.scale(image135, (130, 70))
image_position135 = (345, 320)

script_dir206 = os.path.dirname(os.path.abspath(__file__))
image_path206 = os.path.join(script_dir206, "images\\green.png")
image136 = pygame.image.load(image_path206)
resized_image136 = pygame.transform.scale(image136, (130, 70))
image_position136 = (495, 255)

script_dir207 = os.path.dirname(os.path.abspath(__file__))
image_path207 = os.path.join(script_dir207, "images\\green.png")
image137 = pygame.image.load(image_path207)
resized_image137 = pygame.transform.scale(image137, (130, 70))
image_position137 = (495, 386)

script_dir2077 = os.path.dirname(os.path.abspath(__file__))
image_path2077 = os.path.join(script_dir2077, "images\\green.png")
image1377 = pygame.image.load(image_path2077)
resized_image1377 = pygame.transform.scale(image1377, (130, 70))
image_position1377 = (695, 456)

script_dir2055 = os.path.dirname(os.path.abspath(__file__))
image_path2055 = os.path.join(script_dir2055, "images\\yellow.png")
image1355 = pygame.image.load(image_path2055)
resized_image1355 = pygame.transform.scale(image1355, (130, 70))
image_position1355 = (300, 456)


def Help():
    screen.blit(BG50, (0, 0))
    screen.blit(BG500, (130, 87))

    text_font = get_font2(40)
    text = text_font.render("Help page 1", True, (204, 32, 27))
    screen.blit(text, (460, 7))

    pygame.draw.rect(screen, (119, 166, 83), (448, 64, 285, 4))

    back_button_font = get_font(66)
    back_button_color = (255, 255, 255)
    back_button_rect = pygame.Rect(7, 7, 36, 30)
    pygame.draw.rect(screen, back_button_color, back_button_rect)
    back_text = back_button_font.render("", True, (0, 0, 0))
    screen.blit(back_text, (back_button_rect.x + 3, 564))

    screen.blit(resized_image6, image_position6)

    next_button_font = get_font(66)
    next_button_color = (255, 255, 255)
    next_button_rect = pygame.Rect(1156, 7, 36, 30)
    pygame.draw.rect(screen, next_button_color, next_button_rect)
    next_text = next_button_font.render("", True, (0, 0, 0))
    screen.blit(next_text, (next_button_rect.x + 3, 564))

    screen.blit(resized_image130, image_position130)




    pygame.display.update()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:

                if back_button_rect.collidepoint(event.pos):
                    Start()
                elif next_button_rect.collidepoint(event.pos):
                    Help1()


script_dir1001 = os.path.dirname(os.path.abspath(__file__))
image_path1001 = os.path.join(script_dir1001, "images\\traa.png")
BG501 = pygame.image.load(image_path1001)
BG501 = pygame.transform.scale(BG501, (920, 520))

script_dir405 = os.path.dirname(os.path.abspath(__file__))
image_path405 = os.path.join(script_dir405, "images\\yellow.png")
image236 = pygame.image.load(image_path405)
resized_image236 = pygame.transform.scale(image236, (130, 70))
image_position236 = (770, 78)

script_dir406 = os.path.dirname(os.path.abspath(__file__))
image_path406 = os.path.join(script_dir406, "images\\green2.png")
image237 = pygame.image.load(image_path406)
resized_image237 = pygame.transform.scale(image237, (130, 70))
image_position237 = (895, 180)

script_dir4066 = os.path.dirname(os.path.abspath(__file__))
image_path4066 = os.path.join(script_dir4066, "images\\green.png")
image2376 = pygame.image.load(image_path4066)
resized_image2376 = pygame.transform.scale(image2376, (130, 70))
image_position2376 = (850, 420)

script_dir4067 = os.path.dirname(os.path.abspath(__file__))
image_path4067 = os.path.join(script_dir4067, "images\\green.png")
image2377 = pygame.image.load(image_path4067)
resized_image2377 = pygame.transform.scale(image2377, (130, 70))
image_position2377 = (850, 340)

script_dir4068 = os.path.dirname(os.path.abspath(__file__))
image_path4068 = os.path.join(script_dir4068, "images\\yellow.png")
image2378 = pygame.image.load(image_path4068)
resized_image2378 = pygame.transform.scale(image2378, (130, 70))
image_position2378 = (770, 210)

script_dir4069 = os.path.dirname(os.path.abspath(__file__))
image_path4069 = os.path.join(script_dir4069, "images\\yellow.png")
image2379 = pygame.image.load(image_path4069)
resized_image2379 = pygame.transform.scale(image2379, (130, 70))
image_position2379 = (690, 260)

script_dir4061 = os.path.dirname(os.path.abspath(__file__))
image_path4061 = os.path.join(script_dir4061, "images\\yellow.png")
image2371 = pygame.image.load(image_path4061)
resized_image2371 = pygame.transform.scale(image2371, (130, 70))
image_position2371 = (170, 270)


def Help1():
    screen.blit(BG50, (0, 0))
    screen.blit(BG501, (100, 87))

    text_font = get_font2(40)
    text = text_font.render("Help page 2", True, (204, 32, 27))
    screen.blit(text, (460, 7))

    pygame.draw.rect(screen, (119, 166, 83), (448, 64, 285, 4))
    back_button_font = get_font(66)
    back_button_color = (255, 255, 255)
    back_button_rect = pygame.Rect(7, 7, 36, 30)
    pygame.draw.rect(screen, back_button_color, back_button_rect)
    back_text = back_button_font.render("", True, (0, 0, 0))
    screen.blit(back_text, (back_button_rect.x + 3, 564))

    screen.blit(resized_image6, image_position6)

    next_button_font = get_font(66)
    next_button_color = (255, 255, 255)
    next_button_rect = pygame.Rect(1156, 7, 36, 30)
    pygame.draw.rect(screen, next_button_color, next_button_rect)
    next_text = next_button_font.render("", True, (0, 0, 0))
    screen.blit(next_text, (next_button_rect.x + 3, 564))

    screen.blit(resized_image130, image_position130)



    pygame.display.update()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:

                if back_button_rect.collidepoint(event.pos):
                    Help()
                elif next_button_rect.collidepoint(event.pos):
                    Help2()

script_dir1002 = os.path.dirname(os.path.abspath(__file__))
image_path1002 = os.path.join(script_dir1002, "images\\evv.png")
BG502 = pygame.image.load(image_path1002)
BG502 = pygame.transform.scale(BG502, (890, 520))

script_dir5506 = os.path.dirname(os.path.abspath(__file__))
image_path5506 = os.path.join(script_dir5506, "images\\green2.png")
image2375 = pygame.image.load(image_path5506)
resized_image2375 = pygame.transform.scale(image2375, (130, 70))
image_position2375 = (873, 170)

script_dir5066 = os.path.dirname(os.path.abspath(__file__))
image_path5066 = os.path.join(script_dir5066, "images\\green.png")
image5376 = pygame.image.load(image_path5066)
resized_image5376 = pygame.transform.scale(image5376, (130, 70))
image_position5376 = (850, 415)

script_dir5069 = os.path.dirname(os.path.abspath(__file__))
image_path5069 = os.path.join(script_dir5069, "images\\yellow.png")
image5379 = pygame.image.load(image_path5069)
resized_image5379 = pygame.transform.scale(image5379, (130, 70))
image_position5379 = (740, 306)

script_dir6069 = os.path.dirname(os.path.abspath(__file__))
image_path6069 = os.path.join(script_dir6069, "images\\green1.png")
image6379 = pygame.image.load(image_path6069)
resized_image6379 = pygame.transform.scale(image6379, (130, 70))
image_position6379 = (765, 530)

script_dir7069 = os.path.dirname(os.path.abspath(__file__))
image_path7069 = os.path.join(script_dir7069, "images\\green.png")
image7379 = pygame.image.load(image_path7069)
resized_image7379 = pygame.transform.scale(image7379, (130, 70))
image_position7379 = (845, 265)

def Help2():
    screen.blit(BG50, (0, 0))
    screen.blit(BG502, (140, 87))

    text_font = get_font2(40)
    text = text_font.render("Help page 3", True, (204, 32, 27))
    screen.blit(text, (460, 7))

    pygame.draw.rect(screen, (119, 166, 83), (448, 64, 285, 4))
    back_button_font = get_font(66)
    back_button_color = (255, 255, 255)
    back_button_rect = pygame.Rect(7, 7, 36, 30)
    pygame.draw.rect(screen, back_button_color, back_button_rect)
    back_text = back_button_font.render("", True, (0, 0, 0))
    screen.blit(back_text, (back_button_rect.x + 3, 564))

    screen.blit(resized_image6, image_position6)

    next_button_font = get_font(66)
    next_button_color = (255, 255, 255)
    next_button_rect = pygame.Rect(1156, 7, 36, 30)
    pygame.draw.rect(screen, next_button_color, next_button_rect)
    next_text = next_button_font.render("", True, (0, 0, 0))
    screen.blit(next_text, (next_button_rect.x + 3, 564))

    screen.blit(resized_image130, image_position130)



    pygame.display.update()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:

                if back_button_rect.collidepoint(event.pos):
                    Help1()
                elif next_button_rect.collidepoint(event.pos):
                    Help3()

script_dir1003 = os.path.dirname(os.path.abspath(__file__))
image_path1003 = os.path.join(script_dir1003, "images\\cameratrack.png")
BG55 = pygame.image.load(image_path1003)
BG55 = pygame.transform.scale(BG55, (920, 520))

script_dir50699 = os.path.dirname(os.path.abspath(__file__))
image_path50699 = os.path.join(script_dir50699, "images\\green.png")
image53799 = pygame.image.load(image_path50699)
resized_image53799 = pygame.transform.scale(image53799, (730, 160))
image_position53799 = (310, 180)

def Help3():
    screen.blit(BG50, (0, 0))
    screen.blit(BG55, (140, 100))

    text_font = get_font2(40)
    text = text_font.render("Help page 4", True, (204, 32, 27))
    screen.blit(text, (460, 7))

    pygame.draw.rect(screen, (119, 166, 83), (448, 64, 285, 4))
    back_button_font = get_font(66)
    back_button_color = (255, 255, 255)
    back_button_rect = pygame.Rect(7, 7, 36, 30)
    pygame.draw.rect(screen, back_button_color, back_button_rect)
    back_text = back_button_font.render("", True, (0, 0, 0))
    screen.blit(back_text, (back_button_rect.x + 3, 564))

    screen.blit(resized_image6, image_position6)

    next_button_font = get_font(66)
    next_button_color = (255, 255, 255)
    next_button_rect = pygame.Rect(1156, 7, 36, 30)
    pygame.draw.rect(screen, next_button_color, next_button_rect)
    next_text = next_button_font.render("", True, (0, 0, 0))
    screen.blit(next_text, (next_button_rect.x + 3, 564))

    screen.blit(resized_image130, image_position130)
    screen.blit(resized_image53799, image_position53799)

    text_font = get_font3(19)
    text = text_font.render("To run the track with camera, you should install 'IP Webcam' on your phone.", True, (0, 0, 0))
    screen.blit(text, (354, 199))

    text_font = get_font3(19)
    text = text_font.render("When you open the app and start a server, an IP address will appear on the screen.", True, (0, 0, 0))
    screen.blit(text, (328, 233))

    text_font = get_font3(19)
    text = text_font.render("You should enter this IP address in this text field.", True, (0, 0, 0))
    screen.blit(text, (470, 268))


    pygame.display.update()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:

                if back_button_rect.collidepoint(event.pos):
                    Help1()


def Contact():
    screen.blit(BG4, (0, 0))

    text_font = get_font(120)
    text = text_font.render("Contact Us", True, (204, 32, 27))
    screen.blit(text, (400, 85))

    screen.blit(resized_image11, image_position11)
    screen.blit(resized_image12, image_position12)
    screen.blit(resized_image13, image_position13)

    text_font = get_font2(30)
    text = text_font.render("address", True, (0, 0, 0))
    screen.blit(text, (246, 328))

    text_font = get_font3(25)
    text = text_font.render("BP 98, Jijel 18000", True, (0, 0, 0))
    screen.blit(text, (225, 382))

    text_font = get_font2(30)
    text = text_font.render("phone", True, (0, 0, 0))
    screen.blit(text, (535, 328))

    text_font = get_font3(25)
    text = text_font.render("0672268133", True, (0, 0, 0))
    screen.blit(text, (523, 382))

    text_font = get_font2(30)
    text = text_font.render("email", True, (0, 0, 0))
    screen.blit(text, (834, 328))

    text_font = get_font3(25)
    text = text_font.render("Menmech@gmail.com", True, (0, 0, 0))
    screen.blit(text, (773, 382))

    back_button_font = get_font(66)
    back_button_color = (255, 255, 255)
    back_button_rect = pygame.Rect(7, 7, 36, 30)
    pygame.draw.rect(screen, back_button_color, back_button_rect)
    back_text = back_button_font.render("", True, (0, 0, 0))
    screen.blit(back_text, (back_button_rect.x + 3, 564))

    screen.blit(resized_image6, image_position6)

    pygame.display.update()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:

                if back_button_rect.collidepoint(event.pos):
                    Start()


script_dir20 = os.path.dirname(os.path.abspath(__file__))
image_path20 = os.path.join(script_dir20, "images\\n8.png")
BG5 = pygame.image.load(image_path20)
BG5 = pygame.transform.scale(BG5, (SCREEN_WIDTH, SCREEN_HEIGHT))

script_dir21 = os.path.dirname(os.path.abspath(__file__))
image_path21 = os.path.join(script_dir21, "images\\cursor.png")
image8 = pygame.image.load(image_path21)
resized_image8 = pygame.transform.scale(image8, (54, 50))
image_position8 = (186, 329)

script_dir22 = os.path.dirname(os.path.abspath(__file__))
image_path22 = os.path.join(script_dir22, "images\\loupe.png")
image19 = pygame.image.load(image_path22)
resized_image19 = pygame.transform.scale(image19, (54, 50))
image_position19 = (482, 329)

script_dir23 = os.path.dirname(os.path.abspath(__file__))
image_path23 = os.path.join(script_dir23, "images\\motion-sensor.png")
image25 = pygame.image.load(image_path23)
resized_image25 = pygame.transform.scale(image25, (49, 45))
image_position25 = (806, 329)

import tkinter as tk
from tkinter import filedialog
import subprocess


def Select_video():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mkv")])
    root.destroy()

    return file_path



font = get_font3(26)
font10 = get_font3(24)

options = ['RED', 'GREEN', 'BLUE']
selected_option = None
dropdown_open = False

dropdown_rect = pygame.Rect(510, 463, 115, 43)
options_rects = [pygame.Rect(510, 501 + i * 38, 115, 43) for i in range(len(options))]

options1 = ['Yes', 'No']
selected_option1 = None
dropdown_open1 = False

dropdown_rect1 = pygame.Rect(510, 524, 115, 43)
options_rects1 = [pygame.Rect(510, 562 + i * 38, 115, 43) for i in range(len(options1))]

options2 = ['Show', 'Not Show']
selected_option2 = None
dropdown_open2 = False

dropdown_rect2 = pygame.Rect(910, 463, 146, 43)
options_rects2 = [pygame.Rect(910, 501 + i * 38, 146, 43) for i in range(len(options2))]

options3 = ['Show', 'Not Show']
selected_option3 = None
dropdown_open3 = False

dropdown_rect3 = pygame.Rect(910, 524, 146, 43)
options_rects3 = [pygame.Rect(910, 562 + i * 38, 146, 43) for i in range(len(options3))]


def draw_choice_box():
    pygame.draw.rect(screen, (179, 207, 165), dropdown_rect)
    text_surface = font.render(selected_option, True, (0, 0, 0))
    screen.blit(text_surface, (515, 465))

    if dropdown_open:
        for i, option in enumerate(options):
            pygame.draw.rect(screen, (179, 207, 165), options_rects[i])
            option_surface = font.render(option, True, (0, 0, 0))
            screen.blit(option_surface, (options_rects[i].x + 2, options_rects[i].y + 10))

    if not dropdown_open:
        pygame.draw.rect(screen, (179, 207, 165), dropdown_rect1)
        text_surface1 = font.render(selected_option1, True, (0, 0, 0))
        screen.blit(text_surface1, (515, 526))

        if dropdown_open1:
            for i, option1 in enumerate(options1):
                pygame.draw.rect(screen, (179, 207, 165), options_rects1[i])
                option_surface1 = font.render(option1, True, (0, 0, 0))
                screen.blit(option_surface1, (options_rects1[i].x + 2, options_rects1[i].y + 10))



    pygame.draw.rect(screen, (179, 207, 165), dropdown_rect2)
    text_surface2 = font.render(selected_option2, True, (0, 0, 0))
    screen.blit(text_surface2, (915, 465))

    if dropdown_open2:
        for i, option2 in enumerate(options2):
            pygame.draw.rect(screen, (179, 207, 165), options_rects2[i])
            option_surface2 = font.render(option2, True, (0, 0, 0))
            screen.blit(option_surface2, (options_rects2[i].x + 2, options_rects2[i].y + 10))


script_dir36 = os.path.dirname(os.path.abspath(__file__))
image_path36 = os.path.join(script_dir36, "images\\down.png")
image29 = pygame.image.load(image_path36)
resized_image29 = pygame.transform.scale(image29, (37, 31))
image_position29 = (591, 468)

script_dir37 = os.path.dirname(os.path.abspath(__file__))
image_path37 = os.path.join(script_dir37, "images\\down.png")
image92 = pygame.image.load(image_path37)
resized_image92 = pygame.transform.scale(image92, (37, 31))
image_position92 = (591, 529)

script_dir38 = os.path.dirname(os.path.abspath(__file__))
image_path38 = os.path.join(script_dir38, "images\\down.png")
image93 = pygame.image.load(image_path38)
resized_image93 = pygame.transform.scale(image93, (37, 31))
image_position93 = (1022, 470)

"""script_dir380 = os.path.dirname(os.path.abspath(__file__))
image_path380 = os.path.join(script_dir380, "down.png")
image930 = pygame.image.load(image_path380)
resized_image930 = pygame.transform.scale(image93, (37, 31))
image_position930 = (1022, 528)"""


def save_selected_options():
    with open("selected_options.txt", "w") as file:
        file.write(f"{selected_option}\n")
        file.write(f"{selected_option1}\n")
        file.write(f"{selected_option2}\n")
        file.write(f"{selected_option3}\n")


selected_video = None


def Preview_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    window_width, window_height = 950, 590
    video_width, video_height = 950, 590

    root = tk.Tk()
    root.title("Video")
    root.geometry(f"{window_width}x{window_height}")

    video_label = tk.Label(root)
    video_label.pack()

    global exit_flag
    exit_flag = False

    def exit_callback():
        global exit_flag
        exit_flag = True
        root.quit()

    exit_button = ttk.Button(root, text="Exit", command=exit_callback)
    exit_button.pack()

    frame_count = 0
    speed_factor = 2

    def show_frame():
        global exit_flag
        nonlocal frame_count
        if exit_flag:
            return

        ret, frame = cap.read()
        if not ret:
            return

        if frame_count % speed_factor == 0:
            resized_frame = cv2.resize(frame, (video_width, video_height))

            frame_image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            frame_image = Image.fromarray(frame_image)
            frame_image = ImageTk.PhotoImage(frame_image)

            video_label.config(image=frame_image)
            video_label.image = frame_image

        frame_count += 1
        root.after(25, show_frame)

    show_frame()

    root.mainloop()

    cap.release()
    cv2.destroyAllWindows()


def TrackVideo():
    global dropdown_open, selected_option
    global dropdown_open1, selected_option1
    global dropdown_open2, selected_option2
    global dropdown_open3, selected_option3

    global selected_video

    while True:

        screen.blit(BG5, (0, 0))
        text_font = get_font2(56)
        text = text_font.render("Track from Video", True, (242, 17, 51))
        screen.blit(text, (295, 18))

        pygame.draw.rect(screen, (232, 250, 132), (274, 94, 110, 4))
        pygame.draw.rect(screen, (232, 250, 132), (274, 48, 4, 50))

        pygame.draw.rect(screen, (174, 227, 163), (262, 105, 70, 4))
        pygame.draw.rect(screen, (174, 227, 163), (262, 68, 4, 38))

        pygame.draw.rect(screen, (232, 250, 132), (818, 20, 110, 4))
        pygame.draw.rect(screen, (232, 250, 132), (924, 20, 4, 50))

        pygame.draw.rect(screen, (174, 227, 163), (870, 10, 70, 4))
        pygame.draw.rect(screen, (174, 227, 163), (937, 10, 4, 38))

        text_font = get_font3(38)
        text = text_font.render("Selected Video :", True, (82, 177, 250))
        screen.blit(text, (38, 197))

        pygame.draw.rect(screen, (232, 250, 132), (185, 442, 910, 3))
        pygame.draw.rect(screen, (232, 250, 132), (185, 586, 910, 3))
        pygame.draw.rect(screen, (232, 250, 132), (185, 442, 3, 145))
        pygame.draw.rect(screen, (232, 250, 132), (1094, 442, 3, 145))

        text_font = get_font3(29)
        text = text_font.render("Bounding Box Color :", True, (82, 177, 250))
        screen.blit(text, (220, 463))

        text_font = get_font3(29)
        text = text_font.render("Save Video :", True, (82, 177, 250))
        screen.blit(text, (220, 524))

        text_font = get_font3(29)
        text = text_font.render("Object Class :", True, (82, 177, 250))
        screen.blit(text, (710, 462))

        """text_font = get_font3(29)
        text = text_font.render("Trajectory :", True, (82, 177, 250))
        screen.blit(text, (710, 524))"""

        select_button_font = get_font(78)
        select_button_color = (161, 207, 145)
        select_button_rect = pygame.Rect(170, 315, 240, 75)

        pygame.draw.rect(screen, select_button_color, select_button_rect)
        select_text = select_button_font.render("Select", True, (0, 0, 0))
        pygame.draw.rect(screen, (215, 235, 250), (180, 325, 220, 55))
        screen.blit(select_text, (254, 316))

        screen.blit(resized_image8, image_position8)

        preview_button_font = get_font(78)
        preview_button_color = (161, 207, 145)
        preview_button_rect = pygame.Rect(470, 315, 260, 75)

        pygame.draw.rect(screen, preview_button_color, preview_button_rect)
        preview_text = preview_button_font.render("Preview", True, (0, 0, 0))
        pygame.draw.rect(screen, (215, 235, 250), (480, 325, 240, 55))
        screen.blit(preview_text, (546, 316))

        screen.blit(resized_image19, image_position19)

        track_button_font = get_font(78)
        track_button_color = (161, 207, 145)
        track_button_rect = pygame.Rect(790, 315, 222, 75)

        pygame.draw.rect(screen, track_button_color, track_button_rect)
        track_text = track_button_font.render("Track", True, (0, 0, 0))
        pygame.draw.rect(screen, (215, 235, 250), (800, 325, 202, 55))
        screen.blit(track_text, (870, 317))

        screen.blit(resized_image25, image_position25)

        back_button_font = get_font(66)
        back_button_color = (255, 255, 255)
        back_button_rect = pygame.Rect(7, 7, 36, 30)
        pygame.draw.rect(screen, back_button_color, back_button_rect)
        back_text = back_button_font.render("", True, (0, 0, 0))
        screen.blit(back_text, (back_button_rect.x + 3, 564))

        screen.blit(resized_image6, image_position6)

        exit_button_font = get_font(44)
        exit_button_color = (233, 250, 227)
        exit_button_rect = pygame.Rect(1132, 590, 57, 51)
        pygame.draw.rect(screen, exit_button_color, exit_button_rect)
        exit_text = exit_button_font.render("", True, (0, 0, 0))
        screen.blit(exit_text, (543, 572))

        screen.blit(resized_image16, image_position16)

        pygame.draw.rect(screen, (255, 255, 255), (360, 197, 760, 60))

        draw_choice_box()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:

                if dropdown_rect.collidepoint(event.pos):
                    dropdown_open = not dropdown_open
                elif dropdown_open:
                    for i, rect in enumerate(options_rects):
                        if rect.collidepoint(event.pos):
                            selected_option = options[i]
                            dropdown_open = False
                            break
                    else:
                        dropdown_open = False

                if dropdown_rect1.collidepoint(event.pos):
                    dropdown_open1 = not dropdown_open1
                elif dropdown_open1:
                    for i, rect in enumerate(options_rects1):
                        if rect.collidepoint(event.pos):
                            selected_option1 = options1[i]
                            dropdown_open1 = False
                            break
                    else:
                        dropdown_open1 = False

                if dropdown_rect2.collidepoint(event.pos):
                    dropdown_open2 = not dropdown_open2
                elif dropdown_open2:
                    for i, rect in enumerate(options_rects2):
                        if rect.collidepoint(event.pos):
                            selected_option2 = options2[i]
                            dropdown_open2 = False
                            break
                    else:
                        dropdown_open2 = False

                if dropdown_rect3.collidepoint(event.pos):
                    dropdown_open3 = not dropdown_open3
                elif dropdown_open3:
                    for i, rect in enumerate(options_rects3):
                        if rect.collidepoint(event.pos):
                            selected_option3 = options3[i]
                            dropdown_open3 = False
                            break
                    else:
                        dropdown_open3 = False

                elif select_button_rect.collidepoint(event.pos):
                    selected_video = Select_video()
                    pygame.draw.rect(screen, (255, 255, 255), (360, 197, 760, 60))

                elif preview_button_rect.collidepoint(event.pos):
                    if selected_video:
                        Preview_video(selected_video)
                elif back_button_rect.collidepoint(event.pos):
                    TrackOption()
                elif track_button_rect.collidepoint(event.pos):
                    if selected_video:
                        save_selected_options()

                        #os.chdir('C:/Users/STS/PycharmProjects/INTERFACE')
                        os.chdir(os.getcwd())



                        current_dir = os.getcwd()

                        track_script = os.path.join(current_dir, 'track3.py')
                        yolo_weights = os.path.join(current_dir, 'yolov7.pt')
                        weights = os.path.join(current_dir, 'osnet_x0_25_msmt1760.pt')

                        subprocess.Popen(
                            ['cmd', '/e', 'start', 'cmd.exe', '/K', 'conda', 'activate', 'Interface1', '&&', 'python',
                             track_script, '--yolo-weights', yolo_weights, '--classes', '0',
                             '--Tweights', weights, '--show-vid', '--source', selected_video,
                             '--save-vid', '--conf-thres', '0.15']
                        )



                    else:
                        ok_button_rect = display_message("Please select a video before tracking.")
                        delete_message_box(ok_button_rect)
                elif exit_button_rect.collidepoint(event.pos):
                    pygame.quit()
                    sys.exit()
        if selected_video:
            font = get_font3(26)
            text_surface = font.render(selected_video, True, (0, 0, 0))
            screen.blit(text_surface, (370, 208))

        screen.blit(resized_image29, image_position29)
        if not dropdown_open:
            screen.blit(resized_image92, image_position92)

        screen.blit(resized_image93, image_position93)
        """if not dropdown_open2:
            screen.blit(resized_image930, image_position930)"""

        pygame.display.flip()


script_dir220 = os.path.dirname(os.path.abspath(__file__))
image_path220 = os.path.join(script_dir220, "images\\loupe.png")
image190 = pygame.image.load(image_path220)
resized_image190 = pygame.transform.scale(image190, (54, 50))
image_position190 = (810, 289)

script_dir221 = os.path.dirname(os.path.abspath(__file__))
image_path221 = os.path.join(script_dir221, "images\\play-button.png")
image191 = pygame.image.load(image_path221)
resized_image191 = pygame.transform.scale(image191, (58, 51))
image_position191 = (390, 487)

script_dir222 = os.path.dirname(os.path.abspath(__file__))
image_path222 = os.path.join(script_dir222, "images\\down.png")
image192 = pygame.image.load(image_path222)
resized_image192 = pygame.transform.scale(image192, (64, 67))
image_position192 = (660, 285)

font20 = get_font3(42)
font12 = get_font3(24)

options100 = ['MOT17-04', 'MOT17-09']
selected_option100 = None
dropdown_open100 = False

dropdown_rect100 = pygame.Rect(420, 280, 300, 75)
options_rects100 = [pygame.Rect(420, 343 + i * 56, 300, 75) for i in range(len(options100))]


def draw_choice_box10():
    pygame.draw.rect(screen, (161, 207, 145), dropdown_rect100)
    text_surface100 = font20.render(selected_option100, True, (0, 0, 0))
    screen.blit(text_surface100, (440, 287))

    if dropdown_open100:
        for i, option100 in enumerate(options100):
            pygame.draw.rect(screen, (181, 214, 169), options_rects100[i])
            option_surface100 = font20.render(option100, True, (0, 0, 0))
            screen.blit(option_surface100, (options_rects100[i].x + 2, options_rects100[i].y + 10))


def Option_Selected(selected_option100):
    root = tk.Tk()
    root.withdraw()
    file_path = None

    if selected_option100 == 'MOT17-04':
        file_path = r"MOT1704Preview.mp4"

    if selected_option100 == 'MOT17-09':
        file_path = r"MOT1709Preview.mp4"

    root.destroy()
    return file_path


def Preview_video1(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    window_width, window_height = 950, 590
    video_width, video_height = 950, 590

    root = tk.Tk()
    root.title("Video")
    root.geometry(f"{window_width}x{window_height}")

    video_label = tk.Label(root)
    video_label.pack()

    global exit_flag
    exit_flag = False

    def exit_callback():
        global exit_flag
        exit_flag = True
        root.quit()

    exit_button = ttk.Button(root, text="Exit", command=exit_callback)
    exit_button.pack()

    def show_frame():
        global exit_flag
        if exit_flag:
            return

        ret, frame = cap.read()
        if not ret:
            return

        resized_frame = cv2.resize(frame, (video_width, video_height))

        frame_image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        frame_image = Image.fromarray(frame_image)
        frame_image = ImageTk.PhotoImage(frame_image)

        video_label.config(image=frame_image)
        video_label.image = frame_image

        root.after(25, show_frame)

    show_frame()

    root.mainloop()

    cap.release()
    cv2.destroyAllWindows()


def save_selected_video():
    with open("selected_video.txt", "w") as file:
        file.write(f"{selected_option100}\n")


def DoEvaluation():
    global selected_video
    global dropdown_open100, selected_option100

    while True:

        screen.blit(BG5, (0, 0))
        text_font = get_font2(56)
        text = text_font.render("Evaluation", True, (242, 17, 51))
        screen.blit(text, (430, 18))

        pygame.draw.rect(screen, (232, 250, 132), (374, 94, 110, 4))
        pygame.draw.rect(screen, (232, 250, 132), (374, 48, 4, 50))

        pygame.draw.rect(screen, (174, 227, 163), (362, 105, 70, 4))
        pygame.draw.rect(screen, (174, 227, 163), (362, 68, 4, 38))

        pygame.draw.rect(screen, (232, 250, 132), (718, 20, 110, 4))
        pygame.draw.rect(screen, (232, 250, 132), (824, 20, 4, 50))

        pygame.draw.rect(screen, (174, 227, 163), (770, 10, 70, 4))
        pygame.draw.rect(screen, (174, 227, 163), (837, 10, 4, 38))

        text_font = get_font3(44)
        text = text_font.render("Choose Video :", True, (82, 177, 250))
        screen.blit(text, (58, 277))
        pygame.draw.rect(screen, (174, 227, 163), (91, 343, 226, 4))

        preview_button_font = get_font(78)
        preview_button_color = (161, 207, 145)
        preview_button_rect = pygame.Rect(800, 280, 260, 75)

        pygame.draw.rect(screen, preview_button_color, preview_button_rect)
        preview_text = preview_button_font.render("Preview", True, (0, 0, 0))
        pygame.draw.rect(screen, (215, 235, 250), (810, 290, 240, 55))
        screen.blit(preview_text, (876, 281))

        screen.blit(resized_image190, image_position190)

        eval_button_font = get_font(76)
        eval_button_color = (96, 173, 240)
        eval_button_rect = pygame.Rect(379, 475, 452, 75)

        pygame.draw.rect(screen, eval_button_color, eval_button_rect)
        eval_text = eval_button_font.render("Start Evaluation", True, (0, 0, 0))
        pygame.draw.rect(screen, (215, 235, 250), (389, 485, 432, 55))
        screen.blit(eval_text, (459, 477))

        screen.blit(resized_image191, image_position191)

        back_button_font = get_font(66)
        back_button_color = (255, 255, 255)
        back_button_rect = pygame.Rect(7, 7, 36, 30)
        pygame.draw.rect(screen, back_button_color, back_button_rect)
        back_text = back_button_font.render("", True, (0, 0, 0))
        screen.blit(back_text, (back_button_rect.x + 3, 564))

        screen.blit(resized_image6, image_position6)

        exit_button_font = get_font(44)
        exit_button_color = (233, 250, 227)
        exit_button_rect = pygame.Rect(1132, 590, 57, 51)
        pygame.draw.rect(screen, exit_button_color, exit_button_rect)
        exit_text = exit_button_font.render("", True, (0, 0, 0))
        screen.blit(exit_text, (543, 572))

        screen.blit(resized_image16, image_position16)

        draw_choice_box10()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if dropdown_rect100.collidepoint(event.pos):
                    dropdown_open100 = not dropdown_open100
                elif dropdown_open100:
                    for i, rect in enumerate(options_rects100):
                        if rect.collidepoint(event.pos):
                            selected_option100 = options100[i]
                            dropdown_open100 = False
                            break
                    else:
                        dropdown_open100 = False


                elif preview_button_rect.collidepoint(event.pos):
                    file_path = Option_Selected(selected_option100)
                    if file_path:
                        print('file path---------- ', file_path)
                        Preview_video1(file_path)
                elif back_button_rect.collidepoint(event.pos):
                    TrackOption()
                elif eval_button_rect.collidepoint(event.pos):
                    file_path = Option_Selected(selected_option100)
                    if file_path:
                        save_selected_video()

                        os.chdir(os.getcwd())


                        current_dir = os.getcwd()

                        do_evaluation_script = os.path.join(current_dir, 'Evaluation.py')
                        yolo_weights = os.path.join(current_dir, 'yolov7.pt')
                        weights = os.path.join(current_dir, 'osnet_x0_25_msmt1760.pt')

                        subprocess.Popen(
                            ['cmd', '/e', 'start', 'cmd.exe', '/K', 'conda', 'activate', 'Interface1', '&&', 'python',
                             do_evaluation_script, '--yolo-weights', yolo_weights, '--classes', '0',
                             '--Tweights', weights, '--show-vid', '--source', file_path,
                             '--save-vid', '--conf-thres', '0.15']
                        )




                    else:
                        ok_button_rect = display_message("Please select a video before tracking.")
                        delete_message_box(ok_button_rect)
                elif exit_button_rect.collidepoint(event.pos):
                    pygame.quit()
                    sys.exit()

        screen.blit(resized_image192, image_position192)

        pygame.display.flip()


def display_message(message):
    message_font = get_font3(27)
    message_box_color = (250, 249, 220)
    message_box_rect = pygame.Rect(660, 300, 490, 170)
    pygame.draw.rect(screen, message_box_color, message_box_rect)
    message_text = message_font.render(message, True, (0, 0, 0))
    message_text_rect = message_text.get_rect(center=(905, 370))
    screen.blit(message_text, message_text_rect)

    ok_button_rect = pygame.Rect(855, 410, 100, 50)
    pygame.draw.rect(screen, (161, 207, 145), ok_button_rect)
    ok_button_font = get_font2(30)
    ok_button_text = ok_button_font.render("OK", True, (255, 255, 255))
    ok_button_text_rect = ok_button_text.get_rect(center=ok_button_rect.center)
    screen.blit(ok_button_text, ok_button_text_rect)
    pygame.display.update()

    return ok_button_rect


def display_message2(message):
    message_font = get_font3(27)
    message_box_color = (250, 249, 220)
    message_box_rect = pygame.Rect(345, 300, 495, 170)
    pygame.draw.rect(screen, message_box_color, message_box_rect)
    message_text = message_font.render(message, True, (0, 0, 0))
    message_text_rect = message_text.get_rect(center=(595, 370))
    screen.blit(message_text, message_text_rect)

    ok_button_rect = pygame.Rect(545, 410, 100, 50)
    pygame.draw.rect(screen, (161, 207, 145), ok_button_rect)
    ok_button_font = get_font2(30)
    ok_button_text = ok_button_font.render("OK", True, (255, 255, 255))
    ok_button_text_rect = ok_button_text.get_rect(center=ok_button_rect.center)
    screen.blit(ok_button_text, ok_button_text_rect)
    pygame.display.update()

    return ok_button_rect


def delete_message_box(ok_button_rect):
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if ok_button_rect.collidepoint(event.pos):
                    TrackVideo()
                    pygame.display.update()
                    return


script_dir24 = os.path.dirname(os.path.abspath(__file__))
image_path24 = os.path.join(script_dir24, "images\\c2.jpg")
BG6 = pygame.image.load(image_path24)
BG6 = pygame.transform.scale(BG6, (SCREEN_WIDTH, SCREEN_HEIGHT))

script_dir25 = os.path.dirname(os.path.abspath(__file__))
image_path25 = os.path.join(script_dir25, "images\\exit2.png")
image17 = pygame.image.load(image_path25)
resized_image17 = pygame.transform.scale(image17, (51, 46))
image_position17 = (9, 592)

script_dir26 = os.path.dirname(os.path.abspath(__file__))
image_path26 = os.path.join(script_dir26, "images\\camera.png")
image18 = pygame.image.load(image_path26)
resized_image18 = pygame.transform.scale(image18, (60, 60))
image_position18 = (135, 434)


def display_message_box(message):
    message_font = get_font3(27)
    message_box_color = (165, 213, 250)
    message_box_rect = pygame.Rect(300, 290, 430, 200)
    pygame.draw.rect(screen, message_box_color, message_box_rect)
    message_text = message_font.render(message, True, (0, 0, 0))
    message_text_rect = message_text.get_rect(center=(510, 370))
    screen.blit(message_text, message_text_rect)

    ok_button_rect = pygame.Rect(455, 430, 100, 50)
    pygame.draw.rect(screen, (161, 207, 145), ok_button_rect)
    ok_button_font = get_font2(30)
    ok_button_text = ok_button_font.render("OK", True, (255, 255, 255))
    ok_button_text_rect = ok_button_text.get_rect(center=ok_button_rect.center)
    screen.blit(ok_button_text, ok_button_text_rect)

    pygame.display.update()

    return ok_button_rect


def TrackCamera():
    message_box_displayed = False
    ok_button_rect = None

    screen.blit(BG6, (0, 0))

    text_font = get_font2(60)
    text = text_font.render("Track from Camera", True, (242, 17, 51))
    screen.blit(text, (180, 40))

    text_font = get_font3(41)
    text = text_font.render("IP adress :", True, (211, 237, 145))
    screen.blit(text, (32, 217))

    pygame.draw.rect(screen, (232, 250, 132), (170, 119, 110, 4))
    pygame.draw.rect(screen, (232, 250, 132), (170, 69, 4, 50))

    pygame.draw.rect(screen, (174, 227, 163), (155, 134, 70, 4))
    pygame.draw.rect(screen, (174, 227, 163), (155, 97, 4, 38))

    pygame.draw.rect(screen, (232, 250, 132), (808, 44, 110, 4))
    pygame.draw.rect(screen, (232, 250, 132), (914, 44, 4, 50))

    pygame.draw.rect(screen, (174, 227, 163), (864, 30, 70, 4))
    pygame.draw.rect(screen, (174, 227, 163), (931, 30, 4, 38))

    back_button_font = get_font(66)
    back_button_color = (233, 250, 227)
    back_button_rect = pygame.Rect(7, 7, 36, 30)
    pygame.draw.rect(screen, back_button_color, back_button_rect)
    back_text = back_button_font.render("", True, (0, 0, 0))
    screen.blit(back_text, (back_button_rect.x + 3, 564))

    screen.blit(resized_image6, image_position6)

    exit_button_font = get_font(44)
    exit_button_color = (233, 250, 227)
    exit_button_rect = pygame.Rect(7, 590, 57, 51)
    pygame.draw.rect(screen, exit_button_color, exit_button_rect)
    exit_text = exit_button_font.render("", True, (0, 0, 0))
    screen.blit(exit_text, (543, 572))

    screen.blit(resized_image17, image_position17)

    pygame.draw.rect(screen, (251, 252, 169), (120, 425, 395, 80))
    pygame.draw.rect(screen, (215, 235, 250), (129, 434, 72, 62))

    open_button_font = get_font2(33)
    open_button_color = (161, 207, 145)
    open_button_rect = pygame.Rect(212, 434, 294, 62)

    pygame.draw.rect(screen, open_button_color, open_button_rect)
    open_text = open_button_font.render("Open Camera", True, (0, 0, 0))
    pygame.draw.rect(screen, (215, 235, 250), (219, 442, 279, 45))

    screen.blit(open_text, (226, 442))

    screen.blit(resized_image18, image_position18)

    pygame.draw.rect(screen, (242, 17, 51), (365, 411, 160, 4))
    pygame.draw.rect(screen, (242, 17, 51), (524, 411, 4, 55))
    pygame.draw.rect(screen, (174, 227, 163), (107, 514, 160, 4))
    pygame.draw.rect(screen, (174, 227, 163), (107, 463, 4, 55))

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    text_color = (0, 0, 0)
    text_font1 = get_font3(29)

    input_box = pygame.Rect(65, 282, 510, 60)

    user_text = ' '
    active = False

    def capture_video_and_detect_faces(source):
        video = cv2.VideoCapture(source)
        while True:
            check, frame = video.read()
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        video.release()
        cv2.destroyAllWindows()

    #os.chdir('C:/Users/STS/PycharmProjects/INTERFACE')

    os.chdir(os.getcwd())

    camera_process = None

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if camera_process:
                    camera_process.terminate()
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if input_box.collidepoint(event.pos):
                    active = not active
                else:
                    active = False

                if message_box_displayed and ok_button_rect and ok_button_rect.collidepoint(event.pos):
                    message_box_displayed = False
                    screen.fill((255, 255, 255))
                    TrackCamera()
                elif back_button_rect.collidepoint(event.pos):
                    Tracking()
                elif open_button_rect.collidepoint(event.pos):
                    if user_text != ' ':
                        source = user_text.strip() + "/video"
                        ok_button_rect = display_message_box("Opening Camera.. Please wait..")
                        message_box_displayed = True

                        current_dir = os.getcwd()

                        track_script = os.path.join(current_dir, 'track3.py')
                        yolo_weights = os.path.join(current_dir, 'yolov7.pt')
                        weights = os.path.join(current_dir, 'osnet_x0_25_msmt1760.pt')

                        if camera_process is None:
                            camera_process = subprocess.Popen(
                                ['cmd', '/e', 'start', 'cmd.exe', '/K', 'conda', 'activate', 'Interface1', '&&',
                                 'python',
                                 track_script, '--yolo-weights', yolo_weights, '--classes', '0',
                                 '--Tweights', weights, '--show-vid', '--source', source,
                                 '--save-vid', '--conf-thres', '0.15']
                            )


                    else:
                        source = user_text.strip() + "/video"
                        ok_button_rect = display_message_box("You should entrer the IP adress..")
                        message_box_displayed = True
                elif exit_button_rect.collidepoint(event.pos):
                    running = False
                    if camera_process:
                        camera_process.terminate()
                    pygame.quit()
                    sys.exit()

            elif event.type == pygame.KEYDOWN:
                if active:
                    if event.key == pygame.K_RETURN:
                        print("Texte saisi :", user_text)
                        user_text = ' '
                    elif event.key == pygame.K_BACKSPACE:
                        user_text = user_text[:-1]
                    else:
                        user_text += event.unicode

        pygame.draw.rect(screen, (240, 250, 215), input_box)
        user_text_surfacee = text_font1.render(user_text, True, text_color)
        screen.blit(user_text_surfacee, (input_box.x + 10, input_box.y + 3))

        pygame.display.update()

    if camera_process:
        camera_process.terminate()


script_dir27 = os.path.dirname(os.path.abspath(__file__))
image_path27 = os.path.join(script_dir27, "images\\b10.jpg")
BG7 = pygame.image.load(image_path27)
BG7 = pygame.transform.scale(BG7, (SCREEN_WIDTH, SCREEN_HEIGHT))

script_dir28 = os.path.dirname(os.path.abspath(__file__))
image_path28 = os.path.join(script_dir28, "images\\video-pause-button.png")
image9 = pygame.image.load(image_path28)
resized_image9 = pygame.transform.scale(image9, (30, 25))
image_position9 = (64, 9)

script_dir29 = os.path.dirname(os.path.abspath(__file__))
image_path29 = os.path.join(script_dir29, "images\\exit2.png")
image20 = pygame.image.load(image_path29)
resized_image20 = pygame.transform.scale(image20, (32, 22))
image_position20 = (123, 11)


def Track():
    while True:
        screen.blit(BG7, (0, 0))

        text_font = get_font2(40)
        text = text_font.render("Multi-Object Tracking", True, (200, 252, 192))
        screen.blit(text, (308, 8))

        pygame.draw.rect(screen, (255, 23, 38), (296, 64, 110, 4))
        pygame.draw.rect(screen, (255, 23, 38), (296, 27, 4, 40))

        pygame.draw.rect(screen, (255, 23, 38), (740, 8, 110, 4))
        pygame.draw.rect(screen, (255, 23, 38), (850, 8, 4, 40))

        pygame.draw.rect(screen, (174, 227, 163), (10, 90, 826, 4))
        pygame.draw.rect(screen, (174, 227, 163), (10, 90, 4, 488))
        pygame.draw.rect(screen, (174, 227, 163), (10, 578, 826, 4))
        pygame.draw.rect(screen, (174, 227, 163), (836, 90, 4, 492))

        pygame.draw.rect(screen, (174, 227, 163), (862, 80, 324, 4))
        pygame.draw.rect(screen, (174, 227, 163), (862, 80, 4, 512))
        pygame.draw.rect(screen, (174, 227, 163), (862, 592, 328, 4))
        pygame.draw.rect(screen, (174, 227, 163), (1186, 80, 4, 512))

        # pygame.draw.rect(screen, (255, 251, 227), (161, 591, 274, 53))

        pause_button_font = get_font(55)
        pause_button_color = (233, 250, 227)
        pause_button_rect = pygame.Rect(60, 7, 40, 30)
        pygame.draw.rect(screen, pause_button_color, pause_button_rect)
        pause_text = pause_button_font.render("", True, (0, 0, 0))
        screen.blit(pause_text, (pause_button_rect.x + 57, 592))

        screen.blit(resized_image9, image_position9)

        # pygame.draw.rect(screen, (255, 251, 227), (505, 591, 145, 53))

        exit_button_font = get_font(55)
        exit_button_color = (233, 250, 227)
        exit_button_rect = pygame.Rect(120, 7, 40, 30)
        pygame.draw.rect(screen, exit_button_color, exit_button_rect)
        exit_text = exit_button_font.render("", True, (0, 0, 0))
        screen.blit(exit_text, (566, 592))

        screen.blit(resized_image20, image_position20)

        back_button_font = get_font(66)
        back_button_color = (233, 250, 227)
        back_button_rect = pygame.Rect(7, 7, 36, 30)
        pygame.draw.rect(screen, back_button_color, back_button_rect)
        back_text = back_button_font.render("", True, (0, 0, 0))
        screen.blit(back_text, (back_button_rect.x + 3, 564))

        screen.blit(resized_image6, image_position6)

        text_font = get_font3(24)
        text = text_font.render("Detection model :", True, (255, 255, 255))
        screen.blit(text, (875, 106))

        pygame.draw.rect(screen, (255, 251, 227), (1074, 108, 99, 33))

        text_font = get_font3(26)
        text = text_font.render("Yolo", True, (0, 0, 0))
        screen.blit(text, (1098, 104))

        text_font = get_font3(24)
        text = text_font.render("Object class :", True, (255, 255, 255))
        screen.blit(text, (875, 160))

        pygame.draw.rect(screen, (255, 251, 227), (1074, 160, 99, 33))

        text_font = get_font3(26)
        text = text_font.render("Person", True, (0, 0, 0))
        screen.blit(text, (1082, 156))

        text_font = get_font3(24)
        text = text_font.render("Number of objects", True, (255, 255, 255))
        screen.blit(text, (875, 214))

        text_font = get_font3(24)
        text = text_font.render("tracked :", True, (255, 255, 255))
        screen.blit(text, (938, 241))

        pygame.draw.rect(screen, (255, 251, 227), (1086, 231, 84, 33))

        text_font = get_font3(24)
        text = text_font.render("FPS :", True, (255, 255, 255))
        screen.blit(text, (875, 296))

        pygame.draw.rect(screen, (255, 251, 227), (1086, 296, 84, 33))

        text_font = get_font3(24)
        text = text_font.render("Bounding Box", True, (255, 255, 255))
        screen.blit(text, (875, 350))

        text_font = get_font3(24)
        text = text_font.render("Color :", True, (255, 255, 255))
        screen.blit(text, (913, 377))

        text_font = get_font3(24)
        text = text_font.render("Object class :", True, (255, 255, 255))
        screen.blit(text, (875, 431))

        text_font = get_font3(24)
        text = text_font.render("Save video :", True, (255, 255, 255))
        screen.blit(text, (875, 485))

        text_font = get_font3(24)
        text = text_font.render("Show video :", True, (255, 255, 255))
        screen.blit(text, (875, 539))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:

                if exit_button_rect.collidepoint(event.pos):
                    pygame.quit()
                    sys.exit()
                elif back_button_rect.collidepoint(event.pos):
                    TrackVideo()

        pygame.display.flip()


script_dir30 = os.path.dirname(os.path.abspath(__file__))
image_path30 = os.path.join(script_dir30, "images\\b9.jpg")
BG8 = pygame.image.load(image_path30)
BG8 = pygame.transform.scale(BG8, (SCREEN_WIDTH, SCREEN_HEIGHT))

script_dir32 = os.path.dirname(os.path.abspath(__file__))
image_path32 = os.path.join(script_dir32, "images\\down.png")
image23 = pygame.image.load(image_path32)
resized_image23 = pygame.transform.scale(image23, (37, 31))
image_position23 = (1075, 457)

script_dir34 = os.path.dirname(os.path.abspath(__file__))
image_path34 = os.path.join(script_dir34, "images\\down.png")
image27 = pygame.image.load(image_path34)
resized_image27 = pygame.transform.scale(image27, (37, 31))
image_position27 = (1075, 290)

script_dir35 = os.path.dirname(os.path.abspath(__file__))
image_path35 = os.path.join(script_dir35, "images\\down.png")
image28 = pygame.image.load(image_path35)
resized_image28 = pygame.transform.scale(image28, (37, 31))
image_position28 = (1075, 374)

script_dir33 = os.path.dirname(os.path.abspath(__file__))
image_path33 = os.path.join(script_dir33, "images\\icons8-google-colab-48.png")
image24 = pygame.image.load(image_path33)
resized_image24 = pygame.transform.scale(image24, (37, 31))
image_position24 = (600, 573)

script_dir333 = os.path.dirname(os.path.abspath(__file__))
image_path333 = os.path.join(script_dir333, "images\\portable.png")
image244 = pygame.image.load(image_path333)
resized_image244 = pygame.transform.scale(image244, (37, 31))
image_position244 = (240, 573)

font2 = get_font3(29)

options8 = ['16', '32']
selected_option8 = None
dropdown_open8 = False

dropdown_rect8 = pygame.Rect(984, 451, 129, 41)
options_rects8 = [pygame.Rect(984, 490 + i * 50, 129, 50) for i in range(len(options8))]

options7 = ['0.0001', '0.0003']
selected_option7 = None
dropdown_open7 = False

dropdown_rect7 = pygame.Rect(984, 369, 129, 41)
options_rects7 = [pygame.Rect(984, 408 + i * 50, 129, 50) for i in range(len(options7))]

options6 = ['20', '40', '60']
selected_option6 = None
dropdown_open6 = False

dropdown_rect6 = pygame.Rect(984, 284, 129, 41)
options_rects6 = [pygame.Rect(984, 322 + i * 50, 129, 50) for i in range(len(options6))]


def draw_choice_box2():
    pygame.draw.rect(screen, (186, 222, 182), dropdown_rect8)
    text_surface8 = font2.render(selected_option8, True, (0, 0, 0))
    screen.blit(text_surface8, (1020, 451))

    if dropdown_open8:
        for i, option8 in enumerate(options8):
            pygame.draw.rect(screen, (205, 232, 202), options_rects8[i])
            option_surface8 = font2.render(option8, True, (0, 0, 0))
            screen.blit(option_surface8, (1020, options_rects8[i].y + 8))

    pygame.draw.rect(screen, (186, 222, 182), dropdown_rect7)
    text_surface7 = font2.render(selected_option7, True, (0, 0, 0))
    screen.blit(text_surface7, (991, 370))

    if dropdown_open7:
        for i, option7 in enumerate(options7):
            pygame.draw.rect(screen, (205, 232, 202), options_rects7[i])
            option_surface7 = font2.render(option7, True, (0, 0, 0))
            screen.blit(option_surface7, (991, options_rects7[i].y + 8))

    pygame.draw.rect(screen, (186, 222, 182), dropdown_rect6)
    text_surface6 = font2.render(selected_option6, True, (0, 0, 0))
    screen.blit(text_surface6, (1020, 285))

    if dropdown_open6:
        for i, option6 in enumerate(options6):
            pygame.draw.rect(screen, (205, 232, 202), options_rects6[i])
            option_surface6 = font2.render(option6, True, (0, 0, 0))
            screen.blit(option_surface6, (1020, options_rects6[i].y + 8))


def Training():
    global dropdown_open8, selected_option8
    global dropdown_open7, selected_option7
    global dropdown_open6, selected_option6

    while True:
        screen.blit(BG8, (0, 0))

        text_font = get_font2(47)
        text = text_font.render("Training", True, (242, 17, 51))
        screen.blit(text, (457, 21))

        pygame.draw.rect(screen, (232, 250, 132), (439, 94, 110, 4))
        pygame.draw.rect(screen, (232, 250, 132), (439, 45, 4, 50))

        pygame.draw.rect(screen, (174, 227, 163), (448, 86, 70, 3))
        pygame.draw.rect(screen, (174, 227, 163), (448, 49, 3, 38))

        pygame.draw.rect(screen, (232, 250, 132), (604, 11, 110, 4))
        pygame.draw.rect(screen, (232, 250, 132), (714, 11, 4, 50))

        pygame.draw.rect(screen, (174, 227, 163), (638, 19, 70, 3))
        pygame.draw.rect(screen, (174, 227, 163), (705, 19, 3, 38))

        back_button_font = get_font(66)
        back_button_color = (233, 250, 227)
        back_button_rect = pygame.Rect(7, 7, 36, 30)
        pygame.draw.rect(screen, back_button_color, back_button_rect)
        back_text = back_button_font.render("", True, (0, 0, 0))
        screen.blit(back_text, (back_button_rect.x + 3, 564))

        screen.blit(resized_image6, image_position6)

        exit_button_font = get_font(44)
        exit_button_color = (233, 250, 227)
        exit_button_rect = pygame.Rect(1153, 7, 36, 30)
        pygame.draw.rect(screen, exit_button_color, exit_button_rect)
        exit_text = exit_button_font.render("", True, (0, 0, 0))
        screen.blit(exit_text, (543, 572))

        screen.blit(resized_image162, image_position162)

        pygame.draw.rect(screen, (252, 250, 230), (20, 140, 1159, 385))

        text_font = get_font3(27)
        text = text_font.render("Dataset :", True, (0, 0, 0))
        screen.blit(text, (48, 167))
        pygame.draw.rect(screen, (175, 196, 222), (305, 170, 180, 41))

        text_font = get_font3(28)
        text = text_font.render("Market-Harris", True, (0, 0, 0))
        screen.blit(text, (309, 172))

        text_font = get_font3(27)
        text = text_font.render("Backbone :", True, (0, 0, 0))
        screen.blit(text, (48, 261))
        pygame.draw.rect(screen, (175, 196, 222), (305, 264, 180, 41))

        text_font = get_font3(27)
        text = text_font.render("OSNet", True, (0, 0, 0))
        screen.blit(text, (353, 265))

        text_font = get_font3(27)
        text = text_font.render("Learning rate :", True, (0, 0, 0))
        screen.blit(text, (710, 366))

        text_font = get_font3(27)
        text = text_font.render("Number of epochs :", True, (0, 0, 0))
        screen.blit(text, (710, 284))

        text_font = get_font3(36)
        text = text_font.render("Choose :", True, (126, 161, 103))
        screen.blit(text, (615, 192))
        pygame.draw.rect(screen, (252, 250, 96), (625, 239, 110, 4))

        text_font = get_font3(27)
        text = text_font.render("Batch size :", True, (0, 0, 0))
        screen.blit(text, (710, 448))

        text_font = get_font3(27)
        text = text_font.render("Optimizer type :", True, (0, 0, 0))
        screen.blit(text, (48, 355))
        pygame.draw.rect(screen, (175, 196, 222), (305, 358, 180, 41))

        text_font = get_font3(27)
        text = text_font.render("Adam ", True, (0, 0, 0))
        screen.blit(text, (361, 358))

        text_font = get_font3(27)
        text = text_font.render("Pretrained model :", True, (0, 0, 0))
        screen.blit(text, (48, 449))
        pygame.draw.rect(screen, (175, 196, 222), (306, 452, 180, 41))

        text_font = get_font3(27)
        text = text_font.render("No", True, (0, 0, 0))
        screen.blit(text, (377, 452))

        pygame.draw.rect(screen, (126, 161, 103), (585, 560, 394, 59))

        gc_button_font = get_font2(18)
        gc_button_color = (252, 251, 197)
        gc_button_rect = pygame.Rect(595, 570, 374, 39)

        pygame.draw.rect(screen, gc_button_color, gc_button_rect)
        gc_text = gc_button_font.render("Training with Google Colab", True, (0, 0, 0))
        screen.blit(gc_text, (651, 575))

        screen.blit(resized_image24, image_position24)

        pygame.draw.rect(screen, (126, 161, 103), (225, 560, 271, 59))

        pc_button_font = get_font2(18)
        pc_button_color = (252, 251, 197)
        pc_button_rect = pygame.Rect(235, 570, 251, 39)

        pygame.draw.rect(screen, pc_button_color, pc_button_rect)
        pc_text = pc_button_font.render("Training with PC", True, (0, 0, 0))
        screen.blit(pc_text, (288, 575))

        screen.blit(resized_image244, image_position244)

        draw_choice_box2()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:

                if dropdown_rect8.collidepoint(event.pos):

                    dropdown_open8 = not dropdown_open8
                elif dropdown_open8:
                    for i, rect in enumerate(options_rects8):
                        if rect.collidepoint(event.pos):
                            selected_option8 = options8[i]
                            dropdown_open8 = False
                            break
                    else:
                        dropdown_open8 = False

                if dropdown_rect7.collidepoint(event.pos):
                    dropdown_open7 = not dropdown_open7
                elif dropdown_open7:
                    for i, rect in enumerate(options_rects7):
                        if rect.collidepoint(event.pos):
                            selected_option7 = options7[i]
                            dropdown_open7 = False
                            break
                    else:
                        dropdown_open7 = False

                if dropdown_rect6.collidepoint(event.pos):
                    dropdown_open6 = not dropdown_open6
                elif dropdown_open6:
                    for i, rect in enumerate(options_rects6):
                        if rect.collidepoint(event.pos):
                            selected_option6 = options6[i]
                            dropdown_open6 = False
                            break
                    else:
                        dropdown_open6 = False

                elif exit_button_rect.collidepoint(event.pos):
                    pygame.quit()
                    sys.exit()
                elif back_button_rect.collidepoint(event.pos):
                    Start()
                elif pc_button_rect.collidepoint(event.pos):

                    PCTraining()

        screen.blit(resized_image27, image_position27)
        if not dropdown_open6:
            screen.blit(resized_image28, image_position28)

        if not dropdown_open6 and not dropdown_open7:
            screen.blit(resized_image23, image_position23)

        pygame.display.flip()


def PCTraining():
    batch_file_path = r"commands.bat"
    subprocess.Popen(["start", "cmd", "/k", batch_file_path], shell=True)

main_menu()
