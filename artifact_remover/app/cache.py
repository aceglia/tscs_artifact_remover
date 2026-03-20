import os
import json

class Cache:
    def __init__(self, cache_file):
        self.cache_file = cache_file
        self.cache_dict = {}
        self.cache_initialized = False
        if os.path.exists(self.cache_file):
            self.load_from_cache()

    def remove_cache(self):
        os.remove(self.cache_file)

    def set_to_cache(self, key, value):
        self.cache_dict[key] = value
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache_dict, f)
        self.cache_initialized = True 

    def load_from_cache(self):
        with open(self.cache_file, 'r') as f:
            self.cache_dict = json.load(f)  
        self.cache_initialized = True 

    def get_from_cache(self, key):
        if not self.cache_initialized:
            return None
        return self.cache_dict.get(key, None)





            