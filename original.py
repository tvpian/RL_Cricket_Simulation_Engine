import kivy
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Rectangle
from kivy.graphics import *
from kivy.properties import (
    NumericProperty, ReferenceListProperty, ObjectProperty
)
from kivy.core.window import Window
from kivy.config import Config
kivy.config.Config.set('graphics','resizable', False)
from kivy.clock import Clock



class CrikBat(Widget):
    pass


class CrikBall(Widget):
    pass


class GameWidget(Widget):
    ball = ObjectProperty(None)
    bat = ObjectProperty(None)
    wicket = ObjectProperty(None)

    
    def __init__(self,**kwargs):
        super().__init__(**kwargs)  
        
        with self.canvas:
            pass
            #Color(1, 1, 0)
            #self.player=Rectangle(pos=(350,0),size=(100,50),source="bat.jpg")
            #Rectangle(pos=(380,350),size=(25,25),source="ball.jpg")
            #Color(1, 1, 1)
            #d = 30.
            #Ellipse(pos=(380,350), size=(d, d),source="ball.jpg")
        self._keyboard = Window.request_keyboard(self._on_keyboard_closed,self)
        self._keyboard.bind(on_key_down=self._on_key_down)
    
        #currentx = self.player.pos[0]
        #currenty = self.player.pos[1]
        #currentx = currentx + 1

        print("Entered")
        
        #self.player.pos = (currentx,currenty)
   

    def serve_ball(self):
        self.ball.center = self.center
        self.ball.velocity = Vector(4, 0).rotate(randint(0, 360))

    def update(self, dt):
        self.ball.move()

        # bounce off top and bottom
        if (self.ball.y < 0) or (self.ball.top > self.height):
            self.ball.velocity_y *= -1

        # bounce off left and right
        if (self.ball.x < 0) or (self.ball.right > self.width):
            self.ball.velocity_x *= -1


    


    def _on_key_down(self,keyboard,keycode,text,modifier):
        print("Keyboard pressed")
        print("Text: ",text)
        print("Modifier: ",modifier)
        print("Keyboard: ",keyboard)
        print("Keycode: ",keycode[1])



    def _on_keyboard_closed(self):
        print("Keyboard closed")
        self._keyboard.unbind(on_key_down=self._on_key_down)
        self._keyboard = None

        
class MyApp(App):
    def build(self):
        #Window.size=(500,500)
        game = GameWidget()
        game.serve_ball()
        Clock.schedule_interval(game.update, 1.0 / 60.0)
        return game()


if __name__=="__main__":
    app=MyApp()
    app.run()


