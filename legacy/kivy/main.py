from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from random import randint
from kivy.config import Config
from kivy.clock import Clock
from kivy.uix.button import Button
import numpy as np

check = 1
# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')


class CrikBall(Widget):
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)

    def move(self):
        self.pos = Vector(*self.velocity) + self.pos


class Fielder(Widget):
    def bounce_ball(self, ball):
        if self.collide_widget(ball):
            vx, vy = ball.velocity
            offset_y = (ball.center_y - self.center_y) / (self.height / 2)
            offset_x = (ball.center_x - self.center_x) / (self.width / 2)

            bounced = Vector(-1 * vx, -1 * vy)
            vel = bounced * 0.2
            ball.velocity = vel.x+offset_x, vel.y + offset_y
            ball.velocity = 0, 0
            ball.center = 400, 300
            return True


class CrikBat(Widget):
    angle = NumericProperty(0)

    def bounce_ball(self, ball):
        if self.collide_widget(ball):
            vx, vy = ball.velocity
            offset_y = (ball.center_y - self.center_y) / (self.height / 2)
            offset_x = (ball.center_x - self.center_x) / (self.width / 2)

            bounced = Vector(-1 * vx, -1 * vy)
            vel = bounced * 0.4
            ball.velocity = vel.x+offset_x, vel.y + offset_y
            return True

        '''
        if self.collide_widget(ball):
            vx, vy = ball.velocity
            offset_y = abs(ball.center_y - self.center_y) #/ (self.height / 2)
            offset_x = abs(ball.center_x - self.center_x) #/ (self.height / 2)
            print("Bat:")
            print("Bat center_y: ", self.center_y)
            print("Bat center_x: ", self.center_x)
            print("Bat height: ", self.height)
            print("Bat y-offset: ", ball.center_y - self.center_y)
            print("Bat x-offset: ", ball.center_x - self.center_x)
            if(offset_y>offset_x):
                offset_y = (ball.center_y - self.center_y)/ (self.height / 2)
                bounced = Vector(-1 * vx, vy)
            else:
                offset_x = abs(ball.center_x - self.center_x)/ (self.height / 2)
                bounced = Vector(vx, -1 * vy)
            vel = bounced #* 1.1
            ball.velocity = vel.x, vel.y 
            return True
        '''


class CrikWicket(Widget):

    def bounce_ball(self, ball):
        if self.collide_widget(ball):
            vx, vy = ball.velocity
            offset_y = (ball.center_y - self.center_y) / (self.height / 2)
            offset_x = (ball.center_x - self.center_x) / (self.width / 2)

            bounced = Vector(-1 * vx, -1 * vy)
            vel = bounced * 0
            ball.velocity = vel.x + offset_x, vel.y + offset_y
            ball.velocity = 0, 0
            ball.center = 400, 300
            return True

        '''
        if self.collide_widget(ball):
            vx, vy = ball.velocity
            offset_y = (ball.center_y - self.center_y) #/ (self.height / 2)
            offset_x = (ball.center_x - self.center_x) #/ (self.width / 2)            
            print("Wicket:")
            print("Wicket center_y: ", self.center_y)
            print("Wicket center_x: ", self.center_x)
            print("Wicket height: ", self.height)
            print("Wicket y-offset: ", ball.center_y - self.center_y)
            print("Wicket x-offset: ", ball.center_x - self.center_x)
            if(offset_y>offset_x):
                offset_y = (ball.center_y - self.center_y)/ (self.height / 2)
                bounced = Vector(-1 * vx, vy)
            else:
                offset_x = abs(ball.center_x - self.center_x)/ (self.height / 2)
                bounced = Vector(vx, -1 * vy)
            vel = bounced #* 1.1
            ball.velocity = vel.x, vel.y 
            return True
        '''


class GameWidget(Widget):
    ball = ObjectProperty(None)
    bat = ObjectProperty(None)
    wicket = ObjectProperty(None)
    score = NumericProperty(0)
    ball_counter=0
    scores=[]
    #angle = NumericProperty(270)
    fielder1 = ObjectProperty(None)
    fielder2 = ObjectProperty(None)
    fielder3 = ObjectProperty(None)
    fielder4 = ObjectProperty(None)
    fielder5 = ObjectProperty(None)
    fielder6 = ObjectProperty(None)
    fielder7 = ObjectProperty(None)
    fielder8 = ObjectProperty(None)
    fielder9 = ObjectProperty(None)
    fielder10 = ObjectProperty(None)

    def re_init_ball(self, ball_no=0):
        self.ball.center = 400, 300
        #self.ball.velocity = Vector(self.ball.center_x-randint(self.wicket.center_x-self.wicket.width/5,self.wicket.center_x+self.wicket.width/5), self.ball.center_y-self.wicket.center_x)*0.05
        if ball_no == 0:
            self.ball.velocity = Vector(0, -3).rotate(-10)
        if ball_no == 1:
            self.ball.velocity = Vector(0, -3).rotate(0)
        if ball_no == 2:
            self.ball.velocity = Vector(0, -3).rotate(10)
        print(self.wicket.x)
        print(self.wicket.center_x)

    def serve_ball(self):
        #self.ball.center = self.wicket.center_x,self.wicket.center_y+300
        self.ball.center = 400, 300
        print("Pos: ", self.ball.x, self.ball.y)
        print("Center: ", self.ball.center_x, self.ball.center_y)
        #self.ball.velocity = Vector(4, 0).rotate(randint(-180,0))
        #self.ball.velocity = Vector(self.ball.center_x - randint(self.wicket.x,self.wicket.x+self.wicket.width), self.ball.center_y - self.wicket.center_y)
        #self.ball.velocity = Vector(self.ball.center_x-randint(350,400), self.wicket.center_y-self.ball.center_y)*0.02
        self.ball.velocity = Vector(0, -3).rotate(10)
        print(self.wicket.pos)
        print(self.wicket.center)

    def update(self, dt):
        self.ball.move()

        '''
        print("Self: ",self.x,self.y)
        print("Self.Centre: ",self.center_x,self.center_y)
        print("Self.Ball.Centre: ",self.ball.center_x,self.ball.center_y)
        print("Self.Ball: ",self.ball.x,self.ball.y)
        print("Self.Ball.size: ",self.ball.size,self.ball.width,self.ball.height)
        print("Self.Bat.Centre: ",self.bat.center_x,self.bat.center_y)
        print("Self.Bat: ",self.bat.x,self.bat.y)
        print("Self.Bat.size: ",self.bat.size,self.bat.width,self.bat.height)
        print("Self.Wicket.Centre: ",self.wicket.center_x,self.wicket.center_y)
        print("Self.Wicket: ",self.wicket.x,self.wicket.y)
        print("Self.Wicket.size: ",self.wicket.size,self.wicket.width,self.wicket.height)
        
        '''

        # bounce of paddles
        if(self.bat.bounce_ball(self.ball)):
            self.score = self.score - 5
        if(self.wicket.bounce_ball(self.ball)):
            self.score = self.score + 4
        if(self.fielder1.bounce_ball(self.ball)):
            self.score = self.score + 1
        if(self.fielder2.bounce_ball(self.ball)):
            self.score = self.score + 1
        if(self.fielder3.bounce_ball(self.ball)):
            self.score = self.score + 1
        if(self.fielder4.bounce_ball(self.ball)):
            self.score = self.score + 1
        if(self.fielder5.bounce_ball(self.ball)):
            self.score = self.score + 1
        if(self.fielder6.bounce_ball(self.ball)):
            self.score = self.score + 1
        if(self.fielder7.bounce_ball(self.ball)):
            self.score = self.score + 1
        if(self.fielder8.bounce_ball(self.ball)):
            self.score = self.score + 1
        if(self.fielder9.bounce_ball(self.ball)):
            self.score = self.score + 1
        if(self.fielder10.bounce_ball(self.ball)):
            self.score = self.score + 1

        # bounce off top and bottom
        if (self.ball.y < 0) or (self.ball.top > self.center_y+150):
            self.score = self.score - 4
            #self.ball.velocity_y *= -1
            self.ball.velocity = 0, 0
            self.ball.center = 400, 300

        # bounce off left and right
        if (self.ball.x < 0) or (self.ball.right > self.width):
            self.score = self.score - 4
            #self.ball.velocity_x *= -1
            self.ball.velocity = 0, 0
            self.ball.center = 400, 300
        
        if self.ball.velocity_x==0 and self.ball.velocity_y==0:
            self.bat_init(self.ball_counter)
            self.field_init(self.ball_counter)
            self.re_init_ball(ball_no=np.random.choice([0,1,2]))
            self.scores.append(self.score)
            self.ball_counter += 1
            if self.ball_counter == 6:
                self.ball_counter = 0
                print(self.scores)
                self.scores=[]
            self.score=0

            

        # bounce off left and right
        '''
        if ((self.ball.x > 350) and (self.ball.x < 450) and (self.ball.y < 75) and (self.ball.y > 0)):
            print("HITTTTTTTTT")
            self.ball.velocity_x *= -1
            self.ball.velocity_y *= -1
        '''

    def on_touch_move(self, touch):
        self.bat.center_x = touch.x
        self.bat.center_y = touch.y

    def field_init(self, ball_no=2):
        '''
        self.fielder1.pos = randint(0,100),randint(100,300)
        self.fielder2.pos = randint(200,300),randint(100,300)
        self.fielder3.pos = randint(450,500),randint(100,300)
        self.fielder4.pos = randint(0,100),randint(100,300)
        self.fielder5.pos = randint(200,300),randint(200,300)
        self.fielder6.pos = randint(450,500),randint(200,300)
        self.fielder7.pos = randint(200,300),randint(200,300)
        self.fielder8.pos = randint(500,600),randint(200,300)
        self.fielder9.pos = randint(600,700),randint(200,300)
        self.fielder10.pos = randint(700,800),randint(200,300)
        '''
        if ball_no == 0 or ball_no == 3:
            self.fielder1.pos = 286, 286
            self.fielder2.pos = 274, 332
            self.fielder3.pos = 306, 370
        elif ball_no == 1 or ball_no == 4:
            self.fielder1.pos = 352, 352
            self.fielder2.pos = 386, 386
            self.fielder3.pos = 423, 352
        else:
            self.fielder1.pos = 462, 372
            self.fielder2.pos = 502, 343
            self.fielder3.pos = 480, 286
        #self.fielder10.pos = randint(700,800),randint(200,300)

    def bat_init(self, ball_no=0):
        '''
        self.bat.pos = randint(self.wicket.x,self.wicket.x+self.wicket.width),self.center_y+40
        self.bat.angle = -30
        '''
        if ball_no == 0:
            self.bat.pos = 350, 85
            self.bat.angle = 0
        elif ball_no == 1:
            self.bat.pos = 387.5, 80
            self.bat.angle = 0
        elif ball_no == 2:
            self.bat.pos = 425, 85
            self.bat.angle = 0
        elif ball_no == 3:
            self.bat.pos = 350, 85
            self.bat.angle = 30
        elif ball_no == 4:
            self.bat.angle = int(np.random.choice([-30, 0, 30]))
            if self.bat.angle == int(-30):
                self.bat.pos = 393, 85
            elif self.bat.angle == int(30):
                self.bat.pos = 380, 85
            else:
                self.bat.pos = 387.5, 85
        else:
            self.bat.pos = 425, 85
            self.bat.angle = -30

    def wicket_init(self):
        self.wicket.center = 400.0, 37.5


class MyApp(App):
    def build(self):
        game = GameWidget()
        game.wicket_init()
        game.bat_init()
        game.field_init()
        #game.serve_ball()
        Clock.schedule_interval(game.update, 1.0 / 60.0)
        self.i = 0
        self.j = 0
        self.test = game
        clearbtn = Button(text='clear', size=(50, 50))
        savebtn = Button(text='alter', pos=(50, 0), size=(50, 50))
        loadbtn = Button(text='ball', pos=(100, 0), size=(50, 50))
        clearbtn.bind(on_release=self.clear_score)
        savebtn.bind(on_release=self.save)
        loadbtn.bind(on_release=self.ball)
        game.add_widget(clearbtn)
        game.add_widget(savebtn)
        game.add_widget(loadbtn)
        return game

    def clear_score(self, obj):  # clear button
        print("Clicked")
        self.test.score = 0
        print(self.test.bat.x, self.test.bat.y)
        print(self.test.wicket.x, self.test.wicket.x+self.test.wicket.width)

    def save(self, obj):  # save button
        print("Clicked")
        self.test.field_init()
        self.test.bat_init(self.i)
        self.i += 1
        if self.i == 6:
            self.i = 0

    def ball(self, obj):  # load button
        print("Clicked")
        self.test.score = 0
        self.test.re_init_ball(self.j)
        self.j += 1
        if self.j == 3:
            self.j = 0


if __name__ == '__main__':
    MyApp().run()
