from multiprocessing.connection import Client


class ReceiverRepresentativeDataset:
    """Receiver for receiving the dataset from a sender representative
    dataset."""
    def __init__(self, address: str):
        """Constructor.

        Args:
            address:
                Address of the sender.
        """
        self.connection = Client(address)

    def __call__(self):
        """See the function "__call__" in
        src/edgetpu_pass/dataset/representative.py for more detail on this
        function, and see the function "run_sender" in the same file for
        details on the protocol.
        """
        while True:
            recv = self.connection.recv()
            if recv == 0:
                return
            yield recv
            self.connection.send(0)
