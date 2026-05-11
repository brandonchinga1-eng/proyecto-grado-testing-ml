class Dummy:
    
    def foo(self):
        pass
        
    @classmethod
    def bar(cls):
        pass
        
print(Dummy.foo == Dummy.foo)
print(Dummy.bar == Dummy.bar)        
