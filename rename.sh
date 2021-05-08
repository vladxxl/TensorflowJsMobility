$i=1
gci *.jpg | %{rename-item $_.Fullname ('{0:d3}.jpg' -f $i++) }