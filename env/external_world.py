import datetime
import pprint
import random
import copy

class ExternalWorld():
    def __init__(self, params, time_step):
        """
        Inputs:
            params: dictionary with the following keys:
                date_time: datetime object or string with format '%Y-%m-%d %H:%M:%S' or 'random'
                time_step: integer with the time step in minutes
        """

        self.params = params

        if self.params['init_date_time']['value'] == 'random':                    
            start = datetime.datetime.strptime("2018-01-01 00:00:00", '%Y-%m-%d %H:%M:%S')
            end = datetime.datetime.strptime("2018-12-31 23:59:59", '%Y-%m-%d %H:%M:%S')
            self.date_time = self.generate_random_date(start, end)
        elif isinstance(self.params['init_date_time']['value'], str):
            if 'end_init_date_time' not in self.params.keys() or self.params['end_init_date_time']['value'] == 'None':
                self.date_time = datetime.datetime.strptime(self.params['init_date_time']['value'], '%Y-%m-%d %H:%M:%S')
            else:
                start = datetime.datetime.strptime(self.params['init_date_time']['value'], '%Y-%m-%d %H:%M:%S')
                end = datetime.datetime.strptime(self.params['end_init_date_time']['value'], '%Y-%m-%d %H:%M:%S')
                self.date_time = self.generate_random_date(start, end)
        elif isinstance(self.params['init_date_time']['value'], datetime.datetime):
            if 'end_init_date_time' not in self.params.keys() or self.params['end_init_date_time']['value'] == None:
                self.date_time = self.params['init_date_time']['value']
            else:
                start = self.params['init_date_time']['value']
                end = self.params['end_init_date_time']['value']
                self.date_time = self.generate_random_date(start, end)
        
        elif isinstance(self.params['init_date_time']['value'], list):
            start = random.choice(self.params['init_date_time']['value'])
            self.date_time = datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
        
        else:
            raise ValueError("The init_date_time parameter should be 'random', a string with format '%Y-%m-%d %H:%M:%S' or a datetime object")

        self.time_step = time_step


    def reset(self):
        if self.params['init_date_time']['value'] == 'random':                    
            start = datetime.datetime.strptime("2018-01-01 00:00:00", '%Y-%m-%d %H:%M:%S')
            end = datetime.datetime.strptime("2018-12-31 23:59:59", '%Y-%m-%d %H:%M:%S')
            self.date_time = self.generate_random_date(start, end)
        elif isinstance(self.params['init_date_time']['value'], str):
            if self.params['end_init_date_time']['value'] == 'None':
                self.date_time = datetime.datetime.strptime(self.params['init_date_time'], '%Y-%m-%d %H:%M:%S')
            else:
                start = datetime.datetime.strptime(self.params['init_date_time']['value'], '%Y-%m-%d %H:%M:%S')
                end = datetime.datetime.strptime(self.params['end_init_date_time']['value'], '%Y-%m-%d %H:%M:%S')
                self.date_time = self.generate_random_date(start, end)        
        elif isinstance(self.params['init_date_time']['value'], list):
            start = random.choice(self.params['init_date_time']['value'])
            self.date_time = datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
        else:
            self.date_time = self.params['init_date_time']['value']
        
        observations = self.gather_observations()

        return observations


    def step(self, verbose = False):
        self.date_time += datetime.timedelta(minutes = self.time_step)

        observations = self.gather_observations()
        if verbose:
            pprint.pprint(observations)
            
        return observations
    
    def predict_step(self):
        copy_self = copy.deepcopy(self)    
        observations = copy_self.step()
        return observations

    def gather_observations(self):
        observations = {
            'date_time': self.date_time,
            'time_step': self.time_step,
        }


        return observations
    

    def generate_random_date(self, start, end):
        """
        Generates a random datetime between two dates (start and end)
        Input:
            start: datetime object
            end: datetime object
        """
        delta = end - start
        int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
        random_second = random.randrange(int_delta)
        date_time = start + datetime.timedelta(seconds = random_second)
        return date_time