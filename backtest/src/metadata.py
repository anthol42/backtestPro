


class Metadata:
    def __init__(self, name, description, author, version):
        self.name = name
        self.description = description
        self.author = author
        self.version = version

    def __str__(self):
        return f"Name: {self.name}\n\tDescription: {self.description}\n\tAuthor: {self.author}\n\tVersion: {self.version}"