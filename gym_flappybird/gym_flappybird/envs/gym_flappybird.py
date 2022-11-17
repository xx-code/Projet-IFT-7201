import gym
from gym import spaces
from flappybird import *
import numpy


class FlappyBirdEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def init_all_value(self):
        self.bird = None
        self.pipes = deque()
        
        self.clock = pygame.time.Clock()

        self.frame_clock = 0
        self.score = 0

    def __init__(self, render_mode="human") -> None:

        # Init game
        pygame.init()
        pygame.display.init()
        self.display_surface = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
        pygame.display.set_caption('Pygame Flappy Bird')
        self.score_font = pygame.font.SysFont(None, 32, bold=True)
        self.images = load_images()

        self.init_all_value()

        self.size = numpy.array([WIN_WIDTH, WIN_HEIGHT/2, WIN_HEIGHT/2])

        # we can catch tow information we idea of human player
            # the position of the Bird (x, y)
            # the Pipe front use the top and the bottom, the position of pipe x, and the height of each pair of pipe [x, topHeigth, bottomHeight]
        self.observation_space = spaces.Dict(
            {
                'agent': spaces.Box(0, self.size[0], shape=(1,), dtype=float),
                'pipeFront': spaces.Box(0, self.size, shape=(3,), dtype=float)
            }
        )

        # the action of our bird, if deside to jump or do nothing
        self.action_space = spaces.Discrete(2)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
    
    def __get_obs(self):
        pipeFront = numpy.zeros((3,))
        distance_bird_pipe = self.pipes[0].x - self.bird.x

        for pipe in self.pipes:
            distance = pipe.x - self.bird.x
            if distance_bird_pipe <= distance:
                pipeFront[0], pipeFront[1], pipeFront[2] = pipe.x, pipe.top_pieces, pipe.bottom_pieces
            distance_bird_pipe = distance

        return {
            'agent': numpy.array([self.bird.x, self.bird.y]),
            'pipeFront': pipeFront
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.init_all_value()

        self.bird = Bird(50, int(WIN_HEIGHT/2 - Bird.HEIGHT/2), 2,
                (self.images['bird-wingup'], self.images['bird-wingdown']))

        pp = PipePair(self.images['pipe-end'], self.images['pipe-body'])
        self.pipes.append(pp)

        info = {}
        observation = self.__get_obs()

        return observation, info


    def step(self, action):

        # display FPS of the game 
        self.clock.tick(FPS)
        
        terminated = False

        if action == 1:
            self.bird.msec_to_climb = Bird.CLIMB_DURATION
        
        # collisation effect
        pipe_collision = any(p.collides_with(self.bird) for p in self.pipes)
        if pipe_collision or 0 >= self.bird.y or self.bird.y >= WIN_HEIGHT - Bird.HEIGHT:
            terminated = True

        
        for x in (0, WIN_WIDTH / 2):
            self.display_surface.blit(self.images['background'], (x, 0))

        while self.pipes and not self.pipes[0].visible:
            self.pipes.popleft()

        for p in self.pipes:
            p.update()
            
            self.display_surface.blit(p.image, p.rect)

        self.bird.update()
        self.display_surface.blit(self.bird.image, self.bird.rect)

        reward = -10 if terminated else 1

        # update and display score
        for p in self.pipes:
            if p.x + PipePair.WIDTH < self.bird.x and not p.score_counted:
                reward = 10
                self.score += 1
                p.score_counted = True

        # event of render_mode
        score_surface = self.score_font.render(str(self.score), True, (255, 255, 255))
        score_x = WIN_WIDTH/2 - score_surface.get_width()/2
        self.display_surface.blit(score_surface, (score_x, PipePair.PIECE_HEIGHT))

        pygame.display.flip()
        self.frame_clock += 1

        if not self.frame_clock % msec_to_frames(PipePair.ADD_INTERVAL):
            pp = PipePair(self.images['pipe-end'], self.images['pipe-body'])
            self.pipes.append(pp)

        observation = self.__get_obs()
        info = {}

        return observation, reward, terminated, False, info 
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _render_frame(self):
        
        if self.render_mode == "human":

            self.clock = pygame.time.Clock()
            self.score_font = pygame.font.SysFont(None, 32, bold=True)  # default font
    
    def close(self):
        if self.display_surface is not None:
            pygame.display.quit()
        pygame.quit()
        

env = FlappyBirdEnv(render_mode='human')
obs = env.reset()


while True:
    # Take a random action
    action = env.action_space.sample()
    obs, reward, terminated, _ ,info = env.step(action)

    env.render()

    if terminated:
        break


env.close()